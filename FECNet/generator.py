import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from random import shuffle
from scipy.ndimage import rotate, zoom
import time

class TripletGenerator:

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
                    X = [X1, X2, X3]
                    yield X, y


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
            tmp_list.append(self.path + '\\' + im_name)

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
            
            if not im:
                break

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

    #TODO add augmentations
    def __augment(self, image):

        # Rotate
        ang = np.random.rand() * 30 ## rotate 20 deg
        if np.random.rand() < 0.5:
            image = rotate(image, -ang, order=1, reshape=False)
        else:
            image = rotate(image, ang, order=1, reshape=False)

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

                    

if __name__ == '__main__':
    from generator import TripletGenerator
    from model import faceNet_inceptionv3_model, test_siam_model
    from utils import distances
    from custom_loss import TripletLoss
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf
    from tensorflow.python.keras import backend as K
    import keras


    path = 'C:\\Users\\47450\\Documents\\ResQ Biometrics\\Data sets\\FEC_dataset\\images\\two-class_triplets'
    out_shape = (28, 28)
    delta_trip_loss = 0.3
    embedding_size = 16 # faceNet uses 128, FECNet uses 16.
    batch_size = 32



    # Data Generator 
    #trip_gen = TripletGenerator(path, out_shape = out_shape, batch_size=batch_size)
    #gen = trip_gen.flow_from_dir()
    #data_len = trip_gen.get_data_len()

    #MNIST digits
    G = TripletTestGenerator(out_shape, batch_size)
    gen = G.flow()
    data_len = G.get_data_len()


    # Model
    #model = faceNet_inceptionv3_model(input_shape = out_shape, embedding_size = embedding_size)
    model = test_siam_model(input_shape=out_shape, embedding_size=embedding_size)
    model.summary()


    # Loss
    L = TripletLoss(delta=delta_trip_loss, embedding_size=embedding_size)


    model.compile(loss=L.trip_loss,
                optimizer='adam')


    # Callbacks 
    save_model_path = 'C:\\Users\\47450\\Documents\\ResQ Biometrics\\ResQ-Biometrics-CNN\\FECNet\\Models\\'
    new_model_name = 'MNIST_digits_test_only_weights.h5'
    save_best = keras.callbacks.ModelCheckpoint(save_model_path + new_model_name,
                                                monitor='loss',
                                                verbose=1,
                                                save_best_only=True,
                                                save_weights_only=True,
                                                mode='min',
                                                period=1)

    # Fit
    history = model.fit_generator(gen, steps_per_epoch=data_len/batch_size, epochs=30, shuffle=False, callbacks = [save_best])


    ####### Predict
    #test_trip_gen = TripletGenerator(path, out_shape = out_shape, batch_size=1)
    #test_gen = test_trip_gen.flow_from_dir()
    
    batch_size = 1

    G = TripletTestGenerator(out_shape, batch_size)
    gen = G.flow()
    data_len = G.get_data_len()

    for x,y in gen:
        prediction = model.predict(x, batch_size = batch_size, steps=1)
        distances(np.squeeze(prediction), embedding_size)
        print(y)
        exit()
