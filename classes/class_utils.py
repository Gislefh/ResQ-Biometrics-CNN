import numpy as np
import cv2
import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, GaussianNoise
from scipy import ndimage
import matplotlib.pyplot as plt


def plot_gen(gen):
    for x, y in gen:
        for i in range(x.shape[0]):
            plt.figure(str(y[i]))
            plt.imshow(np.squeeze(x[i]))
            plt.show()


def get_vgg16_from_keras(input_shape, N_classes):
    if input_shape is None:
        inc_top = True
        inp_shape = None

    else:
        inc_top = False
        inp_shape = input_shape

    model = keras.applications.vgg16.VGG16(include_top=inc_top, weights=None, input_tensor=None, input_shape=inp_shape,
                                           pooling=None, classes=N_classes)
    return model


def get_vgg_w_imnet(input_shape, N_classes, show_trainability=True, freeze_layers=True):
    init_model = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=input_shape,
                                                pooling='max', classes=None)

    x = Dense(255, activation='relu')(init_model.output)
    x = Dropout(0.1)(x)
    x = Dense(255, activation='relu')(x)
    x = Dropout(0.1)(x)
    preds = Dense(N_classes, activation='softmax')(x)
    model = Model(init_model.input, preds)

    if freeze_layers:
        ## freeze all but the last 14 layers. last 6 conv2d and the dense layers
        for layer in model.layers[:-14]:
            layer.trainable = False

        print('Showing which layers are trainable:')
        for i, layer in enumerate(model.layers):
            print('layer nr:', i, ', name:', layer.name, ', trainable:', layer.trainable)

    return model


def get_MobileNetV2(input_shape, N_classes):
    init_model = keras.applications.mobilenet_v2.MobileNetV2(input_shape=input_shape, include_top=False,
                                                             weights='imagenet',
                                                             classes=N_classes,
                                                             pooling='max')

    x = Dense(128, activation='relu')(init_model.output)
    preds = Dense(N_classes, activation='softmax')(x)
    model = Model(init_model.input, preds)

    return model


def get_Xception(input_shape, N_classes, freeze_to_layer='all_but_dense'):
    init_model = keras.applications.xception.Xception(include_top=False, weights='imagenet', input_shape=input_shape,
                                                      pooling='max')

    x = Dense(128, activation='relu')(init_model.output)
    x = Dropout(0.4)(x)
    preds = Dense(N_classes, activation='softmax')(x)
    model = Model(init_model.input, preds)

    if freeze_to_layer == 'all_but_dense':
        for layer in model.layers:
            if layer.name == 'global_max_pooling2d_1':
                break
            layer.trainable = False
    else:
        for layer in model.layers:
            if layer.name == freeze_to_layer:
                break
            layer.trainable = False

    return model


def get_inception_w_imnet(input_shape, N_classes, show_trainability=True, freeze_layers=False):
    init_model = keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet',
                                                             input_shape=input_shape,
                                                             pooling='max', classes=None)

    x = Dense(255, activation='relu')(init_model.output)
    x = Dropout(0.3)(x)
    preds = Dense(N_classes, activation='softmax')(init_model.output)
    model = Model(init_model.input, preds)

    ## freeze all but the last n layers
    if freeze_layers:
        for layer in model.layers[:-68]:
            layer.trainable = False
        print('Showing which layers are trainable:')
        for i, layer in enumerate(model.layers):
            print('layer nr:', i, ', name:', layer.name, ', trainable:', layer.trainable)
    return model


def get_denseNet_w_imnet(input_shape, N_classes):
    init_model = keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', input_shape=input_shape,
                                                         pooling='max', classes=None)

    x = Dense(255, activation='relu')(init_model.output)
    x = Dropout(0.3)(x)
    x = Dense(255, activation='relu')(x)
    x = Dropout(0.3)(x)
    preds = Dense(N_classes, activation='softmax')(x)
    model = Model(init_model.input, preds)

    return model


def add_classes_to_model(model_path, N_classes, freeze_N_layers=None):
    model = load_model(model_path)
    model.pop()
    model.add(Dense(N_classes, activation='softmax', name='pred_layer'))

    if freeze_N_layers != None:
        for layer in model.layers[:-(freeze_N_layers + 1)]:
            layer.trainable = False

    return model


def meta_data(model_name, data,
              path_to_folder='C:\\Users\\47450\\Documents\\ResQ Biometrics\\ResQ-Biometrics-CNN\Models\\Tensorboard\\metadata_models\\'):
    file1 = open(path_to_folder + model_name + '.txt', 'w')
    file1.write('Model name: ' + model_name + '\n')
    for key in data.keys():
        file1.write(key + ' : ' + str(data[key]) + '\n')
    file1.close()


def test_model():
    myInput = Input(shape=(96, 96, 3))

    x = Conv2D(32, (4, 4), activation='relu', input_shape=(96, 96, 3))(myInput)
    x = Conv2D(32, (4, 4), activation='relu')(x)
    M = Model(inputs=myInput, outputs=x)
    return M


class DataAug:

    def __init__(self):
        self.aug_method = []

    def get_functions(self):
        return self.aug_method

    # min: the lowest value
    # max: the highest value
    # [min, max] shuld be in the range [0.3, 3], (isj)
    def add_gamma_transform(self, min, max):
        self.aug_method.append(self.__gamma_transfrom)
        self.gamma_transform_args = [min, max]

    # -----
    def __gamma_transfrom(self, image):
        gamma = np.random.uniform(self.gamma_transform_args[0], self.gamma_transform_args[1])
        return np.clip(np.power(self.image, gamma), 0, 1)
