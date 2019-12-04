import sys

sys.path.insert(0, "classes")
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from class_model import SecModel
from class_generator import Generator
from class_utils import get_vgg16_from_keras, get_vgg_w_imnet, add_classes_to_model, meta_data, get_Xception

import keras
from keras.models import load_model
import h5py
import tensorflow as tf
import os

## paths
train_path = 'C:/Users/47450/Documents/ResQ Biometrics/Data sets/ExpW/train'
new_model_name = 'model_expw_preTr_Xcept_3.h5'
save_model_path = 'Models/'

if new_model_name in os.listdir(save_model_path):
    print('Model name exists. Change the model name')
    #exit()

## consts
N_channels = 3
N_images_per_class = 4000
batch_size = 64
image_shape = (128, 128)
N_classes = 6
X_shape = (batch_size, image_shape[0], image_shape[1], N_channels)
Y_shape = (batch_size, N_classes)
val_size = 0.3

### generator
gen_train = Generator(train_path, X_shape, Y_shape, N_classes, N_channels, batch_size, train_val_split=val_size, N_images_per_class=N_images_per_class, class_list=['angry', 'disgust', 'happy', 'neutral', 'sad', 'surprise'])
gen_train.add_rotate(max_abs_angle_deg=30)
gen_train.add_gamma_transform(0.5,1.5)
gen_train.add_flip()
gen_train.add_shift(0.1)
#gen_train.add_zoom(zoom_range= [0.2,2])

train_gen = gen_train.flow_from_mem(set = 'train', fast_aug=True)
val_gen = gen_train.flow_from_mem(set = 'val', augment_validation = True, fast_aug=True)



### -- get new model
# m = SecModel(N_classes)
# model = m.random_CNN(input_shape = (image_shape[0], image_shape[1], N_channels))
# model.summary()

### -- vgg16 + empty
#model = get_vgg_w_imnet((image_shape[0], image_shape[1], N_channels), N_classes)


### Xception
model = get_Xception(input_shape=(image_shape[0], image_shape[1], N_channels), N_classes=N_classes, freeze_to_layer = 'add_7')


### -- fresh vgg
# model = get_vgg16_from_keras((image_shape[0], image_shape[1], N_channels), N_classes)
# model.save(save_model_path + new_model_name)
# model = keras.applications.vgg16.VGG16(include_top=False, weights= None, input_tensor=None, input_shape=(image_shape[0], image_shape[1], N_channels), pooling=None, classes=N_classes)
# exit()

### --- load model
#model = load_model('Models/model_expw_preTr_Xcept_1.h5')

## use pretrained model
# model = add_classes_to_model('Models\\model_9.h5', 3, 10)

## callbacks
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                           min_delta=0,
                                           patience=10,
                                           verbose=0,
                                           mode='auto',
                                           baseline=None,
                                           restore_best_weights=True)

save_best = keras.callbacks.ModelCheckpoint(save_model_path + new_model_name,
                                            monitor='val_loss',
                                            verbose=1,
                                            save_best_only=True,
                                            save_weights_only=False,
                                            mode='min',
                                            period=1)

## --save as date--
tensorboard_name = datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard = keras.callbacks.TensorBoard(log_dir='C:/Users/47450/Documents/ResQ Biometrics/ResQ-Biometrics-CNN/Models/Tensorboard/' + tensorboard_name, 
                                            histogram_freq=0, 
                                            batch_size=batch_size, 
                                            write_graph=True, 
                                            write_grads=True, 
                                            write_images=True, 
                                            embeddings_freq=0, 
                                            embeddings_layer_names=None, 
                                            embeddings_metadata=None, 
                                            embeddings_data=None, 
                                            update_freq='epoch')

callback = [tensorboard, save_best]

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

steps_per_epoch = np.floor(gen_train.get_length_data() * (1 - val_size)) / batch_size
val_steps_per_epoch = np.floor(gen_train.get_length_data() * val_size) / batch_size

## un-freeze some layers at the top of the network
for layer in model.layers:
    layer.trainable = True



history = model.fit_generator(train_gen,
                    validation_data = val_gen,
                    steps_per_epoch = steps_per_epoch, 
                    validation_steps = val_steps_per_epoch,
                    epochs = 100,
                    callbacks = callback,
                    use_multiprocessing = False)
'''
for layer in model.layers:
    if layer.name == 'add_7':
        break
    layer.trainable = False

history = model.fit_generator(train_gen,
                    validation_data = val_gen,
                    steps_per_epoch = steps_per_epoch, 
                    validation_steps = val_steps_per_epoch,
                    epochs = 100,
                    callbacks = callback,
                    use_multiprocessing = False)
'''
meta_data_dict = {'model_name' : new_model_name,
                'train_size' : steps_per_epoch*batch_size,
                'val_size' : val_steps_per_epoch*batch_size,
                'batch_size' : batch_size,
                'train_path' : train_path,
                'model_classes': gen_train.get_classes(),
                'model_augmentations' : gen_train.get_aug(),
                'Images per class' : N_images_per_class,
                'model_input_shape' : X_shape,
                'N_epochs' : len(history.history['val_loss']),
                'best val loss' : min(history.history['val_loss']),
                'model_used' : 'random CNN'
}
meta_data(new_model_name, meta_data_dict)
