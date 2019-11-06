import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from random import shuffle
from scipy.ndimage import rotate, zoom


class TripletGenerator:

    """
    Generator for FECNet dataset. Assumes that the images are all saved in the same folder (for now).
    """

    def __init__(self, path, out_shape, batch_size, augment = False, data = None, train_val_split = 0.3):

        self.path = path                            # Path to folder containing images
        self.triplet_paths = []                     # init trip train paths        
        self.triplet_paths_val = []                 # init trip validation paths
        self.augment = augment                      # if True -> augment the data. else dont
        self.out_shape = out_shape                  # (x, y, challels)
        self.batch_size = batch_size                # batch size
        self.data = data                            # Number of triplets/3
        self.train_val_split = train_val_split      # Size of the validation set

        # Get Triplet Lists
        self.__triplet_list()


    def flow_from_dir(self, set = 'train'):
        if not self.triplet_paths:
            print('No data found')
            return None

        X1 = np.zeros((self.batch_size, self.out_shape[0], self.out_shape[1], self.out_shape[2]), dtype=np.float32)
        X2 = np.zeros((self.batch_size, self.out_shape[0], self.out_shape[1], self.out_shape[2]), dtype=np.float32)
        X3 = np.zeros((self.batch_size, self.out_shape[0], self.out_shape[1], self.out_shape[2]), dtype=np.float32)
        y = np.zeros((self.batch_size), dtype=np.float32)

        if set == 'train':
            paths = self.triplet_paths
        elif set == 'val':
            paths = self.triplet_paths_val


        while True:
            shuffle(paths)
            for i, triplet in enumerate(paths):
                tmp_list = self.__open_images(triplet)

                X1[i%self.batch_size] = tmp_list[0]
                X2[i%self.batch_size] = tmp_list[1]
                X3[i%self.batch_size] = tmp_list[2]
                y[i%self.batch_size] = int(triplet[-1])

                if i%self.batch_size == self.batch_size -1:
                    #X = [X1, X2, X3]
                    yield [X1, X2, X3], y


    def get_data_len(self, set = 'train'):
        if not self.triplet_paths:
            print('No data found')
        if set == 'train':
            return len(self.triplet_paths)
        elif set == 'val':
            return len(self.triplet_paths_val)
        else:
            print('Choose either val or train')

    '''
    -  function assumes the names of the files are saved as: {row_in_cvs}_{image_nr_in_triplet}_{label}.(jpg/png...)
    saves triplets in self.triplet_paths as [path_to_im_1, ..2 , ..3, label]
    '''
    def __triplet_list(self): 
        tmp_list = []
        for i, im_name in enumerate(sorted(os.listdir(self.path))):
            tmp_list.append(self.path + '/' + im_name)

            # Break if data_len is reached
            if self.data:
                if i >= self.data:
                    break

            # Save as val
            if (np.random.rand() <= self.train_val_split) and (i%3 == 2):  ## NOTE the validation set will be random -> each time the class is initiated
                label = (im_name.split('_')[-1]).split('.')[0]
                self.triplet_paths_val.append([tmp_list[0], tmp_list[1], tmp_list[2], label])
                tmp_list = []    

            # Save as train
            elif i%3 == 2:              
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

            # add 1-dim in front
            #im = np.expand_dims(im, 0)
            if im.shape != self.out_shape:
                if im.shape[:2] != self.out_shape[:2]:
                    seq = [self.out_shape[0] / im.shape[0], self.out_shape[1] / im.shape[1], 1]
                    im = zoom(im, seq, order = 1)


            # augment image
            if self.augment: 
                im = self.__augment(im)

            # normalize
            im = np.clip(im/255, 0, 1)


            tmp_list.append(im)

        return tmp_list

    #TODO add more augmentations
    def __augment(self, image):

        # Rotate - rotates +- n deg
        n = 30
        ang = np.random.rand() * n 
        if np.random.rand() < 0.5:
            image = rotate(image, -ang, order=1, reshape=False)
        else:
            image = rotate(image, ang, order=1, reshape=False)

        # Flip - about the horisontal axis
        if np.random.rand() > 0.5:
            image = np.flip(image, axis = 1)

        # Gamma transfrom - range: [min_, max]
        min_, max_ = 0.8, 1.1
        gamma = np.random.uniform(min_, max_)
        image = np.power(image, gamma)


        return image


"""
Test generator to test the new triplet loss function for FECNet
"""
class TripletTestGenerator:

    def __init__(self, image_shape, batch_size, dataset = 'MNIST_DIGITS', data_len = 50000):
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.dataset = dataset
        self.data_len = data_len


        if self.dataset == 'MNIST_DIGITS':
            self.x_orig, self.y_orig = self.__load_mnist_digits()
        else:
            raise Exception('Unknown dataset')

    def flow(self):
        triplets = self.__create_triplets()
        if len(self.image_shape) == 2:     
            X1 = np.zeros((self.batch_size,self.image_shape[0], self.image_shape[1]))
            X2 = np.zeros((self.batch_size,self.image_shape[0], self.image_shape[1]))
            X3 = np.zeros((self.batch_size,self.image_shape[0], self.image_shape[1]))
        elif len(self.image_shape) == 3:
            X1 = np.zeros((self.batch_size,self.image_shape[0], self.image_shape[1], self.image_shape[2]))
            X2 = np.zeros((self.batch_size,self.image_shape[0], self.image_shape[1], self.image_shape[2]))
            X3 = np.zeros((self.batch_size,self.image_shape[0], self.image_shape[1], self.image_shape[2]))
        Y = np.zeros((self.batch_size))

        while True:
            for i, (im1,im2,im3,label) in enumerate(triplets):
                X1[i%self.batch_size] = im1
                X2[i%self.batch_size] = im2
                X3[i%self.batch_size] = im3
                Y[i%self.batch_size] = label

                if i%self.batch_size == self.batch_size -1:
                    X = [X1, X2, X3]
                    yield X, Y
                
    def get_data_len(self):
        return self.data_len

    def __create_triplets(self):
        tmp = []
        for i, old_label in enumerate(self.y_orig):
            if i == self.data_len:
                break

            rand_c = np.random.randint(1,4)
            
            if rand_c == 1:
                im1 = self.x_orig[i]
                im2, im3, label = self.__find_next(old_label, i)
                if label == 0:
                    tmp.append([im1, im2, im3, 3])
                else:
                    tmp.append([im1, im2, im3, 2])

            elif rand_c == 2:
                im2 = self.x_orig[i]
                im1, im3, label = self.__find_next(old_label, i)
                if label == 0:
                    tmp.append([im1, im2, im3, 3])
                else:
                    tmp.append([im1, im2, im3, 1])

            else:
                im3 = self.x_orig[i]
                im1, im2, label = self.__find_next(old_label, i)
                if label == 0:
                    tmp.append([im1, im2, im3, 2])
                else:
                    tmp.append([im1, im2, im3, 1])

        return tmp


    def __load_mnist_digits(self):
        from keras.datasets import mnist
        (x_train, y_train), (_, _) = mnist.load_data()
        return x_train, y_train

    def __find_next(self, old_label, index):
        im1_bool = False
        im2_bool = False
        
        for i in range(index+1,len(self.y_orig)):

            if self.y_orig[i] == old_label and im1_bool == False:
                im1 = self.x_orig[i]
                im1_bool = True

            if self.y_orig[i] != old_label and im2_bool == False:
                im2 = self.x_orig[i]
                im2_bool = True

            if im1_bool and im2_bool:
                if np.random.rand() < 0.5:
                    return im1, im2, 0 
                else:
                    return im2, im1, 1




class TripletFromOtherDataset:
    """
    -- Generator creating triplets from dataset with defined facial expressions classes -- 

    flow_from_dir - ment for training 

    ret_with_label - ment for testing

    path with images aved as:
        train_folder:
            - angry
                - im1.jpg
                - im2.jpg
                ...
            - sad 
                - im1.jpg
                - im2.jpg
                ...
        ...
    """
    def __init__(self, data_path, batch_size, image_shape, augment = True, label_list = [], N_images_per_label = None):
        # Constants
        self.data_path = data_path
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.label_list = label_list
        self.N_images_per_label = N_images_per_label
        self.augment = augment

        # Functions
        self.image_list, self.labels = self.__get_image_paths()


    def get_labels(self):
        return self.labels

    def get_data_len(self):
        return len(self.image_list)

    def ret_with_label(self):
        X1 = np.zeros((1, self.image_shape[0], self.image_shape[1], self.image_shape[2]))
        X2 = np.zeros((1, self.image_shape[0], self.image_shape[1], self.image_shape[2]))
        X3 = np.zeros((1, self.image_shape[0], self.image_shape[1], self.image_shape[2]))
        Y = np.zeros((3))
        while True:
            for index, [im_path, label] in enumerate(self.image_list):
                
                if index%3 == 0:
                    X1[0] = self.__get_image(im_path)
                    Y[index%3] = label
                if index % 3 == 1:
                    X2[0] = self.__get_image(im_path)
                    Y[index%3] = label
                if index % 3 == 2:
                    X3[0] = self.__get_image(im_path)
                    Y[index%3] = label    

                    yield [X1, X2, X3], Y           

    def flow_from_dir(self):
        triplets = self.__find_triplets()

        X1 = np.zeros((self.batch_size,self.image_shape[0], self.image_shape[1], self.image_shape[2]))
        X2 = np.zeros((self.batch_size,self.image_shape[0], self.image_shape[1], self.image_shape[2]))
        X3 = np.zeros((self.batch_size,self.image_shape[0], self.image_shape[1], self.image_shape[2]))
        Y = np.zeros((self.batch_size))

        while True:
            for i, (im1,im2,im3,label) in enumerate(triplets):
                X1[i%self.batch_size] = self.__get_image(im1)
                X2[i%self.batch_size] = self.__get_image(im2)
                X3[i%self.batch_size] = self.__get_image(im3)
                Y[i%self.batch_size] = label

                if i%self.batch_size == self.batch_size -1:
                    X = [X1, X2, X3]
                    yield X, Y

    def __get_image_paths(self):
        image_list = []
        labels = []
        
        folder_index = 0

        for folder in os.listdir(self.data_path):
            if self.label_list:
                if folder not in self.label_list:
                    continue
                else:
                    labels.append(folder)
            else:
                labels.append(folder)
            folder_index += 1

            for image_index, image in enumerate(os.listdir(self.data_path + '/' + folder)):
                if self.N_images_per_label:
                    if image_index > self.N_images_per_label:
                        break
                image_list.append([self.data_path + '/' + folder + '/' + image, folder_index])
        shuffle(image_list)
        return image_list, labels

    def __get_image(self, path):
        im = cv2.imread(path)
        if not np.shape(im):
            print('-- FROM SELF -- No image found in given path')
            exit()
        
        # GRAY to RBG or BGR to RGB
        if len(im.shape) != 3:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        elif path.split('.')[-1] == 'jpg':
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        
        # Reshape
        if im.shape != self.image_shape:		
            factor_x = self.image_shape[0] / im.shape[0]
            factor_y = self.image_shape[1] / im.shape[1]
            im = zoom(im, (factor_x, factor_y, 1), order = 1)
        
        # Augment
        if self.augment:
            # Rotate - rotates +- n deg
            n = 30
            ang = np.random.rand() * n 
            if np.random.rand() < 0.5:
                im = rotate(im, -ang, order=1, reshape=False)
            else:
                im = rotate(im, ang, order=1, reshape=False)

            # Flip - about the horisontal axis
            if np.random.rand() > 0.5:
                im = np.flip(im, axis = 1)

            # Gamma transfrom - range: [min_, max]
            min_, max_ = 0.8, 1.1
            gamma = np.random.uniform(min_, max_)
            im = np.power(im, gamma)
        
        im = np.clip(im/255, 0, 1)
        return im

    def __find_triplets(self):
        trip = []
        for index, [im_path, label] in enumerate(self.image_list):
            if self.N_images_per_label:
                if index >= self.N_images_per_label:
                    break
            im = im_path
            
            rand_c = np.random.randint(1,4)
            
            if rand_c == 1:
                im1 = im
                im2, im3, label = self.__find_next(label, index)
                if label == 0:
                    trip.append([im1, im2, im3, 3])
                elif label == 2:
                    break
                else:
                    trip.append([im1, im2, im3, 2])

            elif rand_c == 2:
                im2 = im
                im1, im3, label = self.__find_next(label, index)
                if label == 0:
                    trip.append([im1, im2, im3, 3])
                elif label == 2:
                    break
                else:
                    trip.append([im1, im2, im3, 1])

            else:
                im3 = im
                im1, im2, label = self.__find_next(label, index)
                if label == 0:
                    trip.append([im1, im2, im3, 2])
                elif label == 2:
                    break
                else:
                    trip.append([im1, im2, im3, 1])
        shuffle(trip)
        return trip

    def __find_next(self, label, index):
        im1_bool = False
        im2_bool = False
        for i in range(index+1,len(self.image_list)):

            if self.image_list[i][1] == label and im1_bool == False:
                im1 = self.image_list[i][0]
                im1_bool = True

            if self.image_list[i][1] != label and im2_bool == False:
                im2 = self.image_list[i][0]
                im2_bool = True

            if im1_bool and im2_bool:
                if np.random.rand() < 0.5:
                    return im1, im2, 0 
                else:
                    return im2, im1, 1
            
        if (not im1_bool) or (not im2_bool): # not found any matches - end search
            return 0, 0, 2 



if __name__ == '__main__':
    data_path = 'C:/Users/47450/Documents/ResQ Biometrics/Data sets/ExpW/train'
    batch_size = 1
    image_shape = (128,128,3)
    T = TripletFromOtherDataset(data_path, batch_size, image_shape, label_list = [], augment = False, N_images_per_label = None)
    gen = T.ret_with_label()
    label_list = T.get_labels()
    for [x1,x2,x3], y in gen:
        plt.imshow(np.squeeze(x1))
        plt.show()
