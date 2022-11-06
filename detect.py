# import the opencv library
import cv2
import dlib
import keyboard
import torch
from scipy.spatial import distance
import numpy as np
import multiprocessing
from playsound import playsound

class Detector:

    def __init__(self):
        f = open("./resources/config.txt", "r")
        self.threshold = float(f.read().strip())
        self.frame_threshold = 12
        self.sleep_frame_count = 0
        self.audio_proc = None
        self.alarm_running = False
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='model.pt', force_reload=True) 
        self.model.conf = 0.30
        self.prediction_incentive = 0 


    def init_audio(self):
        self.audio_proc = multiprocessing.Process(target=playsound, args=('./resources/alert.wav',))


    def calc_dst(self, eye):
        p1 = distance.euclidean(eye[1], eye[5])
        p2 = distance.euclidean(eye[2], eye[4])
        p3 = distance.euclidean(eye[0], eye[3])

        return (p1 + p2) / (2.0 * p3)


    def check_trigger(self):
        if(self.sleep_frame_count >= self.frame_threshold):
            if(not self.alarm_running):
                self.init_audio()
                self.audio_proc.start()
                self.alarm_running = True

        else:
            if(self.audio_proc != None):
                            self.audio_proc.terminate()
                            self.audio_proc = None
            self.alarm_running = False
        


    def detection_loop(self):
        
        vid = cv2.VideoCapture(0)

        face_detector = dlib.get_frontal_face_detector()
        ld_detector = dlib.shape_predictor("model.dat")

        self.sleep_frame_count = 0

        while(True):
            
            # Capture the video frame
            ret, frame = vid.read()
        
            # Convert to grayscale for cascade
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.equalizeHist(frame_gray)

            results = self.model(frame[..., ::-1])
            
            p_results = results.pandas().xyxy[0].to_numpy()

            if(len(p_results) > 0):
                x0,y0,x1,y1,confi,cla,name = p_results[0]
                output = f"Conf:{round(confi, 2)},Class:{cla}, X0={round(x0, 2)},Y0={round(y0, 2)},X1={round(x1, 2)},Y1={round(y1, 2)},{name}\n"
                # print(output)
                if (cla == 1):
                    self.prediction_incentive = 0.2
                elif (cla == 2):
                    self.prediction_incentive = 0.6


            if(ret):
                faces = face_detector(frame_gray)

                for face in faces:
                    x = face.left()
                    y = face.top()
                    w = face.right() - x
                    h = face.bottom() - y
                    frame = cv2.rectangle(frame, (x,y), (x+w, y+h),(0, 150, 255), 2)
    

                    face_landmarks = ld_detector(frame_gray, face)

                    l_eye = []
                    r_eye = []

                    for n in range(36, 42):
                        x = face_landmarks.part(n).x
                        y = face_landmarks.part(n).y
                        l_eye.append((x,y))
                    
                    for n in range(42, 48):
                        x = face_landmarks.part(n).x
                        y = face_landmarks.part(n).y
                        r_eye.append((x,y))
            
                    
                    l_eye_hull = cv2.convexHull(np.array(l_eye))
                    r_eye_hull = cv2.convexHull(np.array(r_eye))

                    cv2.drawContours(frame, [l_eye_hull], 0, (90, 150, 55), -1)
                    cv2.drawContours(frame, [r_eye_hull], 0, (90, 150, 55), -1)

                    dst = (self.calc_dst(l_eye) + self.calc_dst(r_eye)) / 2

                    if(dst < self.threshold + self.prediction_incentive):
                        self.sleep_frame_count+=1
                        self.check_trigger()

                    else:
                        self.sleep_frame_count = 0

            else:
                dst = 404

            # Adjust the threshold. 
            if keyboard.is_pressed("w"):
                self.threshold = self.threshold + 0.01
                print(self.threshold)

            if keyboard.is_pressed("s"):
                self.threshold = self.threshold - 0.01
                print(self.threshold)


     
            cv2.putText(frame, "Dst: {}".format(round(dst, 2)), (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 255), 1)
            cv2.putText(frame, "Threshold: {}".format(round(self.threshold + self.prediction_incentive, 2)), (10, 60), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 255), 1)


            cv2.imshow('frame', frame)
            

            # Q to Exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                f = open("./resources/config.txt", "w")
                f.write(str(self.threshold))
                break
        

        vid.release()
        cv2.destroyAllWindows()





def main():    
    detect = Detector()
    detect.detection_loop()




if __name__ == "__main__":
    main()