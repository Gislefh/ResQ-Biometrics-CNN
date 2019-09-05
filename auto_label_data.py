import sys
sys.path.insert(0, "classes")
import os
import keras
import numpy as np
from scipy import ndimage
import cv2
from keras.models import load_model
from class_generator import Generator
from class_predict import Predict
import matplotlib.pyplot as plt

##const
path_to_google_data_cvs =  'C:\\Users\\47450\\Documents\\ResQ Biometrics\\Data sets\\FEC_dataset\\faceexp-comparison-data-train-public.csv'
save_path = 'C:\\Users\\47450\\Documents\\ResQ Biometrics\Data sets\\automatically labeled facial expressions high conf\\'
N_channels = 1
batch_size = 16
model_shape = (70, 70)
image_shape = (120, 100)
N_classes = 6
X_shape = (batch_size, image_shape[0], image_shape[1], N_channels)
Y_shape = (batch_size, N_classes)

## create ganerator
N_rows_in_csv = int(np.load(save_path + 'metadata.npy'))

gen = Generator(path_to_google_data_cvs, X_shape, Y_shape, N_classes, N_channels, batch_size)
W_gen = gen.face_from_web_gen(start_row = N_rows_in_csv)

model = load_model("Models\\test_model.h5")
label_list = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
label_number = np.zeros((len(label_list)))
cnt = 0
row_counter = N_rows_in_csv
for image in W_gen:
    print('images saved this session: ', cnt, 'total rows: ', row_counter)
    row_counter += 1
    ### rescale
    zoom_fac_x  = model_shape[0] / image_shape[0]
    zoom_fac_y  = model_shape[1] / image_shape[1]
    rescaled_im = ndimage.zoom(image, (zoom_fac_x, zoom_fac_y, 1), order = 3)

    # to gray and range [0,1]
    gray = cv2.cvtColor(rescaled_im, cv2.COLOR_BGR2GRAY)
    gray = np.clip(gray / 255, 0, 1)
    gray = np.expand_dims(gray, 0)
    gray = np.expand_dims(gray, -1)

    pred = model.predict(gray)
    pred_label = label_list[np.argmax(pred)]
    conf = pred[0, np.argmax(pred)]

    if conf > 0.5:
        ## saving 200 of each class
        cnt += 1
        #if label_number[np.argmax(pred)] < 200:
        #    label_number[np.argmax(pred)] += 1
            
        #else:
        #    print(pred_label, 'not saving')
        #    continue



        ### test if the folder exists
        if not pred_label in os.listdir(save_path):
            os.mkdir(save_path + pred_label)
            cv2.imwrite(save_path + pred_label + '\\' + str(cnt)+'.jpg', image)

            
            

        else:
            cv2.imwrite(save_path + pred_label + '\\' + str(cnt)+'.jpg', image)
            ##update metadata
            np.save(save_path + 'metadata.npy', str(row_counter))

    tmp = 0
    for item in label_number:
        if item > 200:
            tmp += 1
    
    if tmp >= len(label_number):
        break



    
        
        

    #plt.figure('label: '+pred_label +'    confidence: '+ str(conf))
    #plt.imshow(orig_im, cmap = 'gray')
    #plt.show()
