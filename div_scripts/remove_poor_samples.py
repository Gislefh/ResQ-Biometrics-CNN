import numpy as np
import matplotlib.pyplot as plt
import cv2
import dlib
import os
from tqdm import tqdm

data_path = 'C:/Users/47450/Documents/ResQ Biometrics/Data sets/ExpW/train'
save_path =  'C:/Users/47450/Documents/ResQ Biometrics/Data sets/ExpW/train_small'

for folder in os.listdir(data_path):

    if folder not in os.listdir(save_path):
        os.mkdir(save_path + '/' + folder)

    for item in tqdm(os.listdir(data_path + '/' + folder)):
        orig_im = cv2.imread(data_path + '/' + folder + '/' + item)
        gray_im = cv2.cvtColor(orig_im, cv2.COLOR_BGR2GRAY)
        detectior = dlib.get_frontal_face_detector()
        detection = detectior(gray_im, 1)
        if len(detection) == 1:
            cv2.imwrite(save_path + '/' + folder + '/' + item, orig_im)
      
