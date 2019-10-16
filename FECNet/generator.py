import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from random import shuffle



class TripletGenerator:

    def __init__(self, path, out_shape, augment = False):
        self.path = path
        self.triplet_paths = []
        self.augment = augment
        self.out_shape = out_shape

    def flow_from_dir(self):
        if not self.triplet_paths:
            self.__triplet_list()

        if not self.triplet_paths:
            print('triplet list is empty')
            return

        while True:
            shuffle(self.triplet_paths)
            for triplet in self.triplet_paths:
                tmp_list = self.__open_images(triplet)
                X = [tmp_list[0], tmp_list[1], tmp_list[2]]
                y = int(triplet[-1])

                yield X, y


    def get_data_len(self):
        if not self.triplet_paths:
            self.__triplet_list()
        return len(self.triplet_paths)

    '''
    -  function assumes the names of the files are saved as: {row_in_cvs}_{image_nr_in_triplet}_{label}.(jpg/png...)

    saves triplets in self.triplet_paths as [path_to_im_1, ..2 , ..3, label]
    '''
    def __triplet_list(self): 
        tmp_list = []
        for i, im_name in enumerate(sorted(os.listdir(self.path))):
            tmp_list.append(self.path + '\\' + im_name)

            if i%3 == 2:
                label = (im_name.split('_')[-1]).split('.')[0]
                self.triplet_paths.append([tmp_list[0], tmp_list[1], tmp_list[2], label])
                tmp_list = []

    '''
    IN
        triplet: list of paths to a triplet
    OUT:
        list of 3 images
    '''
    def __open_images(self, triplet):
        tmp_list = []
        for i in range(3):
            # load image
            im = cv2.imread(triplet[i])

            # if image is .jpg convert to rgb
            if triplet[i].split('.')[-1] == 'jpg':
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            # normalize
            im = np.clip(im/255, 0, 1)

            # add 1-dim in front
            im = np.expand_dims(im, 0)


            # augment image
            if self.augment: 
                im = self.__augment(im)

            

            tmp_list.append(im)

        return tmp_list

    #TODO add augmentations
    def __augment(self, image):
        return image


        
if __name__ == '__main__':
    path = 'C:\\Users\\47450\\Documents\\ResQ Biometrics\\Data sets\\FEC_dataset\\images\\two-class_triplets'
    gen = TripletGenerator(path)
    a = gen.flow_from_dir()

    print(gen.get_data_len())
    exit()

    for x,y in a:
        print(y)
        for i in range(3):
            plt.figure(str(i))
            plt.imshow(x[i])    
        plt.show()
        exit()
        