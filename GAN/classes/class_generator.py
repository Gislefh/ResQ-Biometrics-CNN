""" generates data to train on
"""

import numpy as np
import os
import dlib
import cv2
from scipy import ndimage
class DataGenerator:

    def __init__(self, data_path, landmark_paths):
        self.data_path = data_path
        self.labels = None


        # Functions
        self.image_paths = self.__find_images()

    def flow_from_dir(self, batch_size, image_shape, average_neutral_image, average_expression_faces): # average_expression_faces needs to be in the same order as label
        I_se = np.zeros((batch_size, image_shape[0], image_shape[1], image_shape[2]))
        I_an = np.zeros((batch_size, image_shape[0], image_shape[1], image_shape[2]))
        I_ae = np.zeros((batch_size, image_shape[0], image_shape[1], image_shape[2]))
        label_out = np.zeros((batch_size, len(self.labels)))

        while True:
            for image_name, label in self.image_paths:
                I_se[i%batch_size] = __open_images(path, image_shape)
                I_an[i%batch_size] = average_neutral_image
                I_ae[i%batch_size] = average_expression_faces[label]

                if i%batch_size == batch_size-1:
                    yield I_an, I_se, I_ae



    def __open_images(self, path, image_shape):
        image = cv2.imread(path)

        if path.split('.')[-1] == 'jpg': # To RBG
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RBG)

        if image.shape != image_shape:
            zoom_x = image_shape[0]/image.shape[0]
            zoom_y = image_shape[1]/image.shape[1]
            image = ndimage.zoom(image, (zoom_x, zoom_y, 1), order=1)
        
        return image

        


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