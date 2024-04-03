import torch
import numpy as np
import cv2
import math
from time import time
from ultralytics import YOLO

class ObjectDetection:

    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        self.model = self.load_model()
        self.stop_now = False
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 0.8
        self.org = (50, 50)
        self.fontScale = 1
        self.color = (255, 0, 0)
        self.thickness = 1
        self.tolerance = 50

        self.duration = 0.1
        self.duration_side =0.1
        self.move = 4 
        self.stoppingFlag = 0
        self.tracker = -1
        self.center_coords = (0,0)


    def center_checker(self, img,x):
        h = img.shape[0]
        w = img.shape[1]

        center_coordinates = (int(w/2),int(h/2))

        rad = math.sqrt( (x[0]-center_coordinates[0])**2)
        # print(rad)
        direction = ""
        
        if(rad <= self.tolerance):  
            # image = cv2.putText(img,f"Inside range: {rad}",x,font,cv2.LINE_AA)
            image = cv2.putText(img, f"Inside range: {rad}", x, self.font,self.fontScale, self.color, self.thickness, cv2.LINE_AA)
            move = 1
            i = 'a'
            print("Going Forward")
            # sendKey(i,move,tracker)
            # print("Sent: ",i)
            tracker = move

            return image
        else:
            if( int(x[0]-center_coordinates[0]) > self.tolerance):
                direction = "Right"
                move = 2
                i = 'c'
                print("Going Rightward")
                # sendKey(i,move,tracker)
                # print("Sent: ",i)
                tracker = move

            elif(int(x[0]-center_coordinates[0]) < (center_coordinates[0]-self.tolerance)):
                direction = "Left"
                move = 3
                i = 'd'
                print("Going Leftward")
                # sendKey(i,move,tracker)
                # print("Sent: ",i)
                tracker = move  
            # image = cv2.putText(img,f"Outside range: {rad}, move: {direction}",x, org, font, cv2.LINE_AA)
            image = cv2.putText(img, f"Outside range: {rad}, move: {direction}", x, self.font,self.fontScale, self.color, self.thickness, cv2.LINE_AA)
            return image
        
    def load_model(self):
        model = YOLO("best_switch_pd_a.pt")
        # model = YOLO("best_s_at_500e(92e_142e).pt")
        # model = YOLO("out_switch.pt")
        model.fuse()
        return model

    def predict(self, frame):
        results = self.model(frame)
        return results


    def plot_bboxes(self, results, frame):
        xyxys = []
        confidences = []
        class_ids = []
        names = self.model.names

        # if not results:  # Check if no objects detected
        #     print("No objects detected")
        #     return frame, xyxys, confidences, class_ids
        


        if results:
            for result in results:
                boxes = result.boxes.cpu().numpy()



        boxes = results[0].boxes.cpu().numpy()
        print(boxes.xyxy)
        xyxys.append(boxes.xyxy)
        confidences.append(boxes.conf)
        class_ids.append(boxes.cls)
        resframe = results[0].plot()
        if((boxes.xyxy).shape[0]>0):
            resframe = cv2.rectangle(resframe,(int(boxes.xyxy[0][0]),int(boxes.xyxy[0][1])),(int(boxes.xyxy[0][2]),int(int(boxes.xyxy[0][3]))),(0,255,0),3)
            a = int((boxes.xyxy[0][0] + boxes.xyxy[0][2]) / 2)
            b = int((boxes.xyxy[0][1] + boxes.xyxy[0][3]) / 2)
            print("Center Coordinates:", a, b)
            resframe = cv2.circle(resframe,(a,b), 5, (0,0,255), -1)
            # for i in range(boxes.xyxy.shape[0]):
                # resframe = cv2.rectangle(resframe,(int(boxes.xyxy[i][0]),int(boxes.xyxy[i][1])),(int(boxes.xyxy[i][2]),int(int(boxes.xyxy[i][3]))),(0,255,0),3)
            self.center_coords = (a,b)
        return resframe, xyxys, confidences, class_ids



    def run(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened(), "Cannot open webcam"

        while self.stop_now is False:
            start_time = time()
            ret, frame = cap.read()
            if not ret:
                break  

            results = self.predict(frame)

            frame, _, confidences, class_ids = self.plot_bboxes(results, frame)

            frame = self.center_checker(frame, self.center_coords)
            cv2.imshow("Object Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            end_time = time()
            inference_time = end_time - start_time
            fps = 1 / inference_time
            print("FPS:", fps)

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = ObjectDetection(0)
    detector.run()