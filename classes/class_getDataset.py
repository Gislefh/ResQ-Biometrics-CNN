from datetime import datetime
import os
import numpy as np
import keras
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import ndimage
from class_utils import one_hot
from class_model import SecModel
from random import shuffle

from pytictoc import TicToc
import csv


class GetDataset:

    def __init__(self, path, X_shape, N_classes, N_channels, train_val_split=0.3, class_list=[],
                 N_images_per_class=None):

        self.path = path  # path to folder, str
        # shape of output, ( width, height, channel) or (width, height, channels)
        self.X_shape = X_shape
        # self.Y_shape = Y_shape  # shape of ground truth, (samples, classes) for classification
        self.N_classes = N_classes  # number of classes, int
        # number of channels in the images, 1 is gray, 3 is color
        self.N_channels = N_channels
        # self.batch_size = batch_size    #number of samples in a batch
        self.image = None
        self.aug_method = []
        self.aug_args = []
        self.train_val_split = train_val_split  # defaults to 0.3
        self.N_images_per_class = N_images_per_class

        if class_list:
            self.class_list = class_list
        else:
            self.class_list = []

    def __from_dir(self, N_images_per_class):

        image_list = []
        class_ = 0

        tmp_val_set = []
        tmp_train_set = []

        for folder in os.listdir(self.path):
            cnt_img_per_class = 0
            if self.class_list:
                if folder not in self.class_list:
                    continue

            # saving N_val first images as validation
            if N_images_per_class != None:
                N_val = int(N_images_per_class * self.train_val_split)
            else:
                N_val = int(len(os.listdir(self.path + '/' + folder))
                            * self.train_val_split)

            for image_ in os.listdir(self.path + '/' + folder):
                if N_images_per_class != None:
                    if cnt_img_per_class > N_images_per_class:
                        break  # TODO fix

                if cnt_img_per_class <= N_val:
                    tmp_val_set.append(
                        [self.path + '/' + folder + '/' + image_, class_])
                else:
                    tmp_train_set.append(
                        [self.path + '/' + folder + '/' + image_, class_])
                cnt_img_per_class += 1

            class_ += 1

        self.val_set = np.array(tmp_val_set)
        self.train_set = np.array(tmp_train_set)

    ''' Creates a generator for either training set or validation set
    - IN:
    set: either val, train or test. str
    N_images_per_class: how many images to get per class
    train_val_split: how much of the data thats used as validaion
    '''

    def flow_from_dir(self, set='train', augment_validation=True):
        if set == 'test':
            self.train_val_split = 0

        self.__from_dir(self.N_images_per_class)

        self.X = np.zeros(
            (len(self.train_set), self.X_shape[0], self.X_shape[1], self.X_shape[2]), np.float32)
        self.Y = np.zeros((len(self.train_set), self.N_classes), np.int8)

        if set == 'train':
            tot_list = self.train_set
        elif set == 'val':
            tot_list = self.val_set
        elif set == 'test':
            tot_list = self.train_set
        else:
            print("select either: 'train', 'val' or 'test'")
            exit()

        index_of_element = list(range(len(tot_list)))
        shuffle(index_of_element)

        print("Collecting image data .....")
        for i in tqdm(range(len(tot_list))):

            choice = index_of_element[i]
            orig_ch = cv2.imread(tot_list[choice, 0]).shape[-1]
            label = int(tot_list[choice, 1])

            if (orig_ch == 3) and (self.N_channels == 1):
                im_tmp = cv2.imread(tot_list[choice, 0])
                self.image = np.expand_dims(cv2.cvtColor(
                    im_tmp, cv2.COLOR_BGR2GRAY), axis=-1)
            else:
                self.image = cv2.imread(tot_list[choice, 0])[
                    :, :, 0:self.N_channels]

            # normalize image to [0,1]
            self.image = np.clip(self.image / 255, 0, 1)

            # reshape image
            if self.image.shape != self.X[0].shape:
                self.X[i] = self.__im_reshape(self.image.shape, self.image)
            else:
                self.X[i] = self.image

            # one hot encode ground truth
            if self.class_list:
                self.Y[i] = one_hot(label, len(self.class_list))
            else:
                self.Y[i] = one_hot(label, self.N_classes)

        return self.X, self.Y




    def __im_reshape(self, orig_shape, image):
        factor_x = self.X_shape[0] / orig_shape[0]
        factor_y = self.X_shape[1] / orig_shape[1]
        return ndimage.zoom(image, (factor_x, factor_y, 1), order=1)

    def get_classes(self):
        if self.class_list:
            return slef.class_list
        else:
            class_list = []
            for folder in os.listdir(self.path):
                class_list.append(folder)
            return class_list


if __name__ == "__main__":
    new_model_name = 'model_test_1.h5'
    save_model_path = 'C:\\Github\\ResQ\\ResQ-Biometrics-CNN\\Models\\'

    # avoid saving over existing model
    if new_model_name in os.listdir(save_model_path):
        print('Model name exists. Change the model name')
        exit()

    # consts
    N_channels = 3
    N_images_per_class = 2000
    image_shape = (80, 80)
    N_classes = 7
    X_shape = (image_shape[0], image_shape[1], N_channels)
    val_size = 0.3

    data_class = GetDataset('C:\\Users\\Eier\\Desktop\\ResQ Dataset\\ExpW\\train',
                            X_shape,
                            N_classes,
                            N_channels,
                            N_images_per_class=N_images_per_class,
                            )

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               min_delta=0,
                                               patience=10,
                                               verbose=0,
                                               mode='auto',
                                               baseline=None,
                                               restore_best_weights=True)

    X, Y = data_class.flow_from_dir()

    m = SecModel(N_classes)
    model = m.random_CNN(input_shape=(
        image_shape[0], image_shape[1], N_channels))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])

    tensorboard_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = keras.callbacks.TensorBoard(
        log_dir='C:\\Github\\ResQ\\ResQ-Biometrics-CNN\\Models\\Tensorboard\\' + tensorboard_name,
        histogram_freq=1,
        write_graph=True,
        write_grads=True,
        write_images=True,
        embeddings_freq=0,
        embeddings_layer_names=None,
        embeddings_metadata=None,
        embeddings_data=None,
        update_freq='epoch')

    save_best = keras.callbacks.ModelCheckpoint(save_model_path + new_model_name,
                                                monitor='val_loss',
                                                verbose=1,
                                                save_best_only=True,
                                                save_weights_only=False,
                                                mode='min',
                                                period=1)

    callback = [tensorboard, save_best, early_stop]
    log = model.fit(x=X, y=Y, epochs=1000, verbose=1, batch_size=64, callbacks=callback, validation_split=0.3,
                    shuffle=True)
