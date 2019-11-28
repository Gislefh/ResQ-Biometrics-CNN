""" generates data to train on
"""

import numpy as np
import os
import dlib
import cv2

class DataGenerator:

    def __init__(self, data_path, landmark_paths):
        self.data_path = data_path
        self.labels = None


        # Functions
        self.image_paths = self.__find_images()

    def image_to_generator(self, batch_size, image_shape, average_neutral_image):
        X = np.zeros((image_shape[0], image_shape[1], image_shape[2]*2))

        while True:
            for i in self.image_paths:
                X[i%batch_size] = 



        


    def get_labels(self):
        return self.labels


    def __find_images(self):
        image_path_list = []
        for folder_cnt, folder in enumerate(os.listdir(self.data_path)):
            self.labels.append([folder, folder_cnt])

            for image_name in os.listdir(self.data_path +'/'+ folder):
                image_path_list.append([self.data_path +'/'+ folder +'/'+ image_name, folder_cnt])

        return image_path_list
                        







if __name__ == '__main__':
    path = 'C:/Users/47450/Documents/ResQ Biometrics/Data sets/ExpW/train_small'
    save_averages_path = 'C:/Users/47450/Documents/ResQ Biometrics/Data sets/IF-GAN_averages/test'
    dlib_class_path = 'C:/Users/47450/Documents/ResQ Biometrics/ResQ-Biometrics-CNN/GAN/dlib_classifier/shape_predictor_68_face_landmarks.dat'
    DG = DataGenerator(path)
    DG.create_averages(save_averages_path, dlib_class_path)