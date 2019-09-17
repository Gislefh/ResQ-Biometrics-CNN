import sys
sys.path.insert(0, "classes")
import tensorflow as tf
import keras
import numpy as np
import cv2
from keras.models import load_model
from class_generator import Generator
from class_predict import Predict
import matplotlib.pyplot as plt

### consts
#test_path = 'C:\\Users\\47450\\Documents\\ResQ Biometrics\\Data sets\\face-expression-recognition-dataset\\images\\validation'
#from_web_path = 'C:\\Users\\47450\\Documents\\ResQ Biometrics\\Data sets\\Data_set_from_web'
#path_to_google_data_cvs =  'C:\\Users\\47450\\Documents\\ResQ Biometrics\\Data sets\\FEC_dataset\\faceexp-comparison-data-train-public.csv'
test_path = 'C:\\Users\\47450\\Documents\\ResQ Biometrics\\Data sets\\ExpW\\validation'

N_channels = 1
batch_size = 16
model_shape_shape = (48, 48)
N_classes = 2
X_shape = (batch_size, model_shape_shape[0], model_shape_shape[1], N_channels)
Y_shape = (batch_size, N_classes)

## create ganerator
gen_test = Generator(test_path, X_shape, Y_shape, N_classes, N_channels, batch_size, N_images_per_class=100, class_list = ['happy', 'neutral'])
N_data = gen_test.get_length_data()
test_gen = gen_test.flow_from_dir(set = 'test')

labels = gen_test.get_classes()

model = load_model("Models\\model_9.h5")
P = Predict(model, labels = labels)

#P.pred_from_cam()
P.conf_matrix(test_gen, N_data)




