import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from imutils.video import VideoStream
import dlib
import time
import pandas as pd
import sys

class Predict:


    def __init__(self, model, labels):
        self.model = model
        self.input_shape = model.inputs[0].shape
        self.output_shape = model.outputs
        self.labels = labels


        #TODO fix this
        #if len(labels) != len(self.output_shape):
        #    print('not the same number of labes as the model')



    #  -- Prints the confution matrix -- 
    # left - the pred 
    # top - ground truth
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
        

    def show_wrongly_labeled(self, generator):
        for x, y in generator:
            for i in range(x.shape[0]):
                pred = self.model.predict(np.expand_dims(x[i], 0))
                pred = np.argmax(pred)
                gt = np.argmax(y[i])
                if gt != pred:
                    plt.figure('label: {}, prediction: {}'.format(self.labels[gt], self.labels[pred]))
                    plt.imshow(np.squeeze(x[i]))
                    plt.show()
       
    def pred_from_cam(self):
        vs = VideoStream(src=0).start()
        time.sleep(0.1)
        frame = vs.read()
        cnt=0
        if self.model.input_shape[-1] == 3:
            N_channels = 3
        else:
            N_channels = 1

        fd = dlib.get_frontal_face_detector()


        while True:
            frame = vs.read()
            
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            try:
                face = fd(gray_image, 1)[0]
                frame_to_model = frame[face.top():face.bottom(), face.left():face.right()]
                cnt = 0
            except:
                cnt +=1
                print('no face detected for ', cnt, 'frames')
                continue

            if frame.shape != self.input_shape[1:4]:
                zoom_factor = [int(self.input_shape[1])/frame_to_model.shape[0], int(self.input_shape[2])/frame_to_model.shape[1]] 
                frame_to_model = ndimage.zoom(frame_to_model, [zoom_factor[0], zoom_factor[1], 1], order =3)

            if N_channels == 1:
                frame_to_model = cv2.cvtColor(frame_to_model, cv2.COLOR_BGR2GRAY)
                frame_to_model = np.expand_dims(frame_to_model, -1)

            frame_to_model = np.expand_dims(frame_to_model, 0)
            

            frame_to_model = frame_to_model/255
            frame_to_model = np.clip(frame_to_model, 0, 1)

            prediction = self.model.predict(frame_to_model)

            prediction_str = self.labels[np.argmax(prediction)]

            conf = prediction[0, np.argmax(prediction)]

            cv2.putText(frame, prediction_str, (50,100), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 0, 0))
            cv2.putText(frame, 'conf: ' +str(conf), (100,200), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 0, 0))
            #cv2.namedWindow('Cam output',cv2.WINDOW_NORMAL)
            #cv2.resizeWindow('Cam output', 400,400)
            cv2.imshow("Cam output", frame)

            cv2.namedWindow('Model input',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Model input', 200,200)
            if N_channels == 1:
                cv2.imshow('Model input', np.squeeze(frame_to_model, (0, -1)))
            else:
                cv2.imshow('Model input', np.squeeze(frame_to_model, 0))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cv2.destroyAllWindows()
        vs.stop()



