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

        try:
            self.model.load_weights(weights_path)
        except:
            print('Downloading config/yolov3.weights')
            os.system("cd config; ./download_weights.sh;")
            self.model.load_weights(weights_path)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(device) # auto select
        # self.model.cuda()
        self.model.eval()

        self.classes = utils.load_classes(class_path)
        self.Tensor  = torch.cuda.FloatTensor

        # folder management
        initial_path  = os.getcwd() 
        self.recorder_path = initial_path + "/recorder/"
        self.directory(self.recorder_path)

        folder_cnt = len(os.walk(self.recorder_path).__next__()[1])
        if folder_cnt >= 1:
            if len(os.walk(self.recorder_path + "cap" + str(folder_cnt-1)).__next__()[2]) == 0:
                folder_cnt -= 1

        self.folder_cap = "cap" + str(folder_cnt)
        self.directory(self.recorder_path + self.folder_cap)

        self.record     = False
        self.video_file = False
        self.fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    def directory(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        
    def detect_image(self, img):
        # scale and pad image
        ratio = min(self.img_size/img.size[0], self.img_size/img.size[1])
        imw   = round(img.size[0] * ratio)
        imh   = round(img.size[1] * ratio)
        img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
            transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                            (128,128,128)), transforms.ToTensor(), ])
        # convert image to self.Tensor
        image_tensor = img_transforms(img).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input_img    = Variable(image_tensor.type(self.Tensor))
        # run inference on the self.model and get detections
        with torch.no_grad():
            detections = self.model(input_img)
            detections = utils.non_max_suppression(detections, 80, self.conf_thres, self.nms_thres)
        return detections[0]

    def crop_object(self, x1, x2, y1, y2, box_w, box_h):
        if x1 < 0:    
            x1_lim = 0
        else:
            x1_lim = x1

        if x1+box_w > self.frame_width:
            x2_lim = self.frame_width
        else:
            x2_lim = x1+box_w
                                
        if y1 < 0:
            y1_lim = 0
        else:
            y1_lim = y1

        if y1+box_h > self.frame_height:
            y2_lim = self.frame_height
        else:
            y2_lim = y1+box_h
        
        return x1_lim, x2_lim, y1_lim, y2_lim

    def run(self):
        cap         = cv2.VideoCapture(0)
        ret, frame  = cap.read()
        self.frame_width  = frame.shape[1]
        self.frame_height = frame.shape[0]
        print ("Video size", self.frame_width,self.frame_height)

        mot_tracker = Sort() 
        frames      = 0
        frame_cnt   = 0
        start_time  = time.time()
        colors      = [(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]

        while(True):
            ret, frame = cap.read()
            if not ret:
                break

            res_frame = frame.copy()

            if self.record:
                self.directory( self.recorder_path + self.folder_cap + '/frames' )

                frame_cnt += 1
                elapsed_time = time.time() - start_time
                cv2.putText(res_frame, 'Rec: ' + str(int(elapsed_time)), (20, 55), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0, 0, 255), lineType=cv2.LINE_AA)

                out.write(frame)
                if frame_cnt%5 == 0:
                    cv2.imwrite(self.recorder_path + self.folder_cap + "/frames/" + str(self.folder_cap) + "-" + str(frame_cnt) + ".jpg", frame)

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
                    y1    = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                    x1    = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
                    color = colors[int(obj_id) % len(colors)]
                    cls   = self.classes[ int(cls_pred) ]

                    cv2.rectangle(res_frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)
                    cv2.rectangle(res_frame, (x1, y1-35), (x1+len(cls)*19+80, y1), color, -1)
                    cv2.putText(res_frame, cls + "-" + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

                    if self.record and frame_cnt%5 == 0:
                        x1_lim, x2_lim, y1_lim, y2_lim = self.crop_object(x1, x2, y1, y2, box_w, box_h)
                        crop = frame[ y1_lim:y2_lim, x1_lim:x2_lim]

                        self.directory( self.recorder_path + self.folder_cap + "/" + str(cls) )

                        file_count = len(os.walk(self.recorder_path + self.folder_cap + "/" + str(cls) + "/").__next__()[2])
                        cv2.imwrite(self.recorder_path    + self.folder_cap      + "/"  + str(cls)    + "/" +
                                    self.folder_cap       + "-" + str(frame_cnt) + 
                                    "-" + str(file_count) + "(" + str(x1_lim)    + ","  + str(y1_lim) +
                                    "," + str(x2_lim)     + "," + str(y2_lim)    + ")"  + ".jpg", crop)

            cv2.putText(res_frame, 'Scenes: ' + self.folder_cap, (20, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (255, 0, 0), lineType=cv2.LINE_AA)
            cv2.imshow('Stream', res_frame)

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                self.record = False
                break

            elif ch == ord('r'):
                if not self.record:
                    if not self.video_file:
                        out = cv2.VideoWriter(self.recorder_path + self.folder_cap + '/' + self.folder_cap +'.avi', \
                              self.fourcc, 30, (self.frame_width, self.frame_height))
                        self.video_file = True
                        print('Start Recording')
                    else:
                        print('Continue Recording')

                    self.record = True
                else:
                    print('Recording is already start..')

            elif ch == ord('t'):
                if self.record:
                    self.record = False
                    print('Stop Recording')
                else:
                    print('Not recording..')

            elif ch == ord('s'):
                if not self.record:
                    details_path = self.recorder_path + self.folder_cap + '/details'
                    self.directory(details_path)

                    photo_no = len(os.walk(details_path).__next__()[2]) + 1
                    cv2.imwrite(details_path + "/" + str(photo_no) + ".jpg", frame)
                    print('Saved :{}'.format(details_path + "/" + str(photo_no) + ".jpg"))
                else:
                    print('Please stop recording first, before saving')

        total_time = time.time() - start_time
        print("{} frames {:.2f} s/frame".format(frames, total_time/frames))
        cv2.destroyAllWindows()
        cap.release()

if __name__ == "__main__":
    yolo = YoloV3()
    yolo.run()