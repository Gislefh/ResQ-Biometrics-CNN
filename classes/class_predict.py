import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from imutils.video import VideoStream
import time
import pandas as pd
import sys

class Predict:


    def __init__(self, model, labels):
        self.model = model
        self.input_shape = model.inputs[0].shape
        self.output_shape = model.outputs
        self.labels = labels

        if len(labels) != len(self.output_shape):
            print('not the same number of labes as the model')



    #  -- Prints the confution matrix -- 
    # left - what it should have guessed
    # top - what it guessed
    def conf_matrix(self, generator, N_images, append_labels = True):
        conf_matrix = np.zeros((len(self.labels), len(self.labels)))
        cnt = 0
        for x, y in generator:
            for i in range(x.shape[0]):
                pred = self.model.predict(np.expand_dims(x[i], 0))
                pred = np.argmax(pred)
                gt = np.argmax(y[i])
                conf_matrix[pred, gt] = conf_matrix[pred, gt] +1
                cnt += 1
            print(cnt,'/', N_images)
                
            if cnt > N_images:
                break

        df = pd.DataFrame(conf_matrix, index = self.labels, columns = self.labels)
        print(df)
        

        


    def pred_from_cam(self):
        vs = VideoStream(src=0).start()
        time.sleep(2.0)
        frame = vs.read()
        


        if frame.shape != self.input_shape[1:4]:
            zoom = True
            zoom_factor = [int(self.input_shape[1])/frame.shape[0], int(self.input_shape[2])/frame.shape[1]] 
        else:
            zoom = False
        
        if frame.shape[-1] != self.input_shape[-1]:
            c2g = True
        else:
            c2g = False
        
        while True:
            frame = vs.read()
            
            if zoom:
                frame_to_model = ndimage.zoom(frame, [zoom_factor[0], zoom_factor[1], 1], order =3)
            else:
                frame_to_model = frame
            if c2g:
                frame_to_model = cv2.cvtColor(frame_to_model, cv2.COLOR_BGR2GRAY)

            frame_to_model = np.expand_dims(frame_to_model, 0)
            frame_to_model = np.expand_dims(frame_to_model, -1)

            frame_to_model = frame_to_model/255
            frame_to_model = np.clip(frame_to_model, 0, 1)

            prediction = self.model.predict(frame_to_model)

            prediction_str = self.labels[np.argmax(prediction)]

            conf = prediction[0, np.argmax(prediction)]

            cv2.putText(frame, prediction_str, (100,100), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 0, 0))
            cv2.putText(frame, 'conf: ' +str(conf), (100,200), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 0, 0))
            cv2.imshow("Frame", frame)

            cv2.namedWindow('f2m',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('f2m', 200,200)
            cv2.imshow('f2m', np.squeeze(frame_to_model, (0, -1)))
           

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cv2.destroyAllWindows()
        vs.stop()



