#!/usr/bin/env python3

from models import *
from utils import *
from sort import *

import os, sys, time, datetime, random
import torch, cv2
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

class YoloV3:
    def __init__(self):
        # load weights and set defaults
        config_path  = 'config/yolov3.cfg'
        weights_path = 'config/yolov3.weights'
        class_path   = 'config/coco.names'
        self.img_size   = 416
        self.conf_thres = 0.8
        self.nms_thres  = 0.4

        # load self.model and put into eval mode
        self.model = Darknet(config_path, img_size=self.img_size)
        self.model.load_weights(weights_path)
        self.model.cuda()
        self.model.eval()

        self.classes = utils.load_classes(class_path)
        self.Tensor  = torch.cuda.FloatTensor

    def detect_image(self, img):
        # scale and pad image
        ratio = min(self.img_size/img.size[0], self.img_size/img.size[1])
        imw = round(img.size[0] * ratio)
        imh = round(img.size[1] * ratio)
        img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
            transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                            (128,128,128)), transforms.ToTensor(), ])
        # convert image to self.Tensor
        image_tensor = img_transforms(img).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input_img = Variable(image_tensor.type(self.Tensor))
        # run inference on the self.model and get detections
        with torch.no_grad():
            detections = self.model(input_img)
            detections = utils.non_max_suppression(detections, 80, self.conf_thres, self.nms_thres)
        return detections[0]

    def run(self):
        colors       = [(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]
        cap          = cv2.VideoCapture(0)
        ret, frame   = cap.read()
        frame_width  = frame.shape[1]
        frame_height = frame.shape[0]
        print ("Video size", frame_width,frame_height)

        mot_tracker  = Sort() 
        frames       = 0
        start_time   = time.time()

        while(True):
            ret, frame = cap.read()
            if not ret:
                break

            frames     += 1
            frame      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pilimg     = Image.fromarray(frame)
            detections = self.detect_image(pilimg)

            frame   = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            img     = np.array(pilimg)
            pad_x   = max(img.shape[0] - img.shape[1], 0) * (self.img_size / max(img.shape))
            pad_y   = max(img.shape[1] - img.shape[0], 0) * (self.img_size / max(img.shape))
            unpad_h = self.img_size - pad_y
            unpad_w = self.img_size - pad_x

            if detections is not None:
                tracked_objects = mot_tracker.update(detections.cpu())

                unique_labels = detections[:, -1].cpu().unique()
                n_cls_preds = len(unique_labels)
                for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
                    box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                    box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
                    y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                    x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
                    color = colors[int(obj_id) % len(colors)]
                    cls = self.classes[int(cls_pred)]
                    cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)
                    cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+80, y1), color, -1)
                    cv2.putText(frame, cls + "-" + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

            cv2.imshow('Stream', frame)
            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break

        total_time = time.time()-start_time
        print("{} frames {} s/frame".format(frames, total_time/frames))
        cv2.destroyAllWindows()
        cap.release()

if __name__ == "__main__":
    yolo = YoloV3()
    yolo.run()