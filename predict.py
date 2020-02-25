### no gpu?
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
####

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
test_path_ferCh = 'C:\\Users\\47450\\Documents\\ResQ Biometrics\\Data sets\\face-expression-recognition-dataset\\images\\validation'
test_path_expw = 'C:/ML/Dataset/ExpW_zip/ExpW/'

N_channels = 3
N_images_per_class = None
batch_size = 1
image_shape = (72 * 3, 64 * 3)
N_classes = 7
X_shape = (batch_size, image_shape[0], image_shape[1], N_channels)
Y_shape = (batch_size, N_classes)
val_size = 0

## create ganerator

gen_test = Generator(test_path_expw, X_shape, Y_shape, N_classes, N_channels, batch_size, train_val_split=val_size,
                      N_images_per_class=N_images_per_class)

N_data = gen_test.get_length_data()
test_gen = gen_test.flow_from_dir(set='test')

labels = gen_test.get_classes()

model = load_model("C:/ML/Models/XceptionV3_expw_0.h5")
save_path = "C:\\ML\\Dataset\\expw_improved\\"

names = os.listdir(test_path_expw)
for i, (x, y) in enumerate(test_gen):
    pred = model.predict(x)
    if np.argmax(pred) == y:
        plt.imsave(save_path + "/" + names[y] + "/", str(i) + ".jpg")

    if i == N_data:
        break
