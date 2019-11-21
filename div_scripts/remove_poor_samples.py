import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pytictoc import TicToc
from tqdm import tqdm

import dlib


def remove_with_dlib():
    '''
    Removes poor samples from the ExpW dataset using the dlib face detector 
    '''


    data_path = 'C:/Users/47450/Documents/ResQ Biometrics/Data sets/ExpW/train'
    save_path =  'C:/Users/47450/Documents/ResQ Biometrics/Data sets/ExpW/train_small'
    T = TicToc()
    
    for folder in os.listdir(data_path):  
        if folder != 'surprise':
            continue

        if folder not in os.listdir(save_path):
            os.mkdir(save_path + '/' + folder)
        
        for item in tqdm(os.listdir(data_path + '/' + folder)):
            if item in os.listdir(save_path + '/' + folder): # pick off where we left off  
                continue

            elif item not in os.listdir(save_path + '/' + folder):
                orig_im = cv2.imread(data_path + '/' + folder + '/' + item)
                gray_im = cv2.cvtColor(orig_im, cv2.COLOR_BGR2GRAY)
                detectior = dlib.get_frontal_face_detector()
                detection = detectior(gray_im, 1)
                if len(detection) == 1:
                    cv2.imwrite(save_path + '/' + folder + '/' + item, orig_im)
            else: 
                print('what?')
            

#def remove_with_model():

  
if __name__ == '__main__':
    remove_with_dlib()
