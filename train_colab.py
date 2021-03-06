"""
Script for colab 
"""

import sys
sys.path.insert(0, "classes")

import numpy as np
import matplotlib.pyplot as plt

from class_model import SecModel
from class_generator import Generator
from class_utils import get_vgg16_from_keras, get_vgg_w_imnet

import keras
from keras.models import load_model
import h5py
import os

## colab spesific
train_path = '/data/ExpW'
save_model_path = '/content/drive/My Drive/ResQBiometrics/Models/'
new_model_name = 'model_17_09_2'
#load_model_path = '/content/drive/My Drive/ResQBiometrics/Models/model_17_09_v2.h5'

N_channels = 3
N_images_per_class = 3500
batch_size = 32
image_shape = (100, 100)
N_classes = 3
X_shape = (batch_size, image_shape[0], image_shape[1], N_channels)
Y_shape = (batch_size, N_classes)
val_size = 0.3

gen_train = Generator(train_path, X_shape, Y_shape, N_classes, N_channels, batch_size, class_list=['happy', 'neutral', 'angry'], train_val_split = val_size, N_images_per_class=N_images_per_class)
gen_train.add_rotate(max_abs_angle_deg=20)
gen_train.add_gamma_transform(0.5,1.5)
gen_train.add_flip()
gen_train.add_shift(0.1)
#gen_train.add_zoom(zoom_range= [0.2,2])

train_gen = gen_train.flow_from_dir(set = 'train')
val_gen = gen_train.flow_from_dir(set = 'val', augment_validation = True)


### -- get model
m = SecModel(N_classes)
model = m.random_CNN(input_shape = (image_shape[0], image_shape[1], N_channels))

### -- vgg16 + empty 
#model = get_vgg_w_imnet((image_shape[0], image_shape[1], N_channels), N_classes)

### --- load model
#model = load_model(load_model_path)

model.summary()

## callbacks
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                min_delta=0, 
                                                patience=5, 
                                                verbose=0, 
                                                mode='auto', 
                                                baseline=None, 
                                                restore_best_weights=True)

save_best = keras.callbacks.ModelCheckpoint(save_model_path + new_model_name + '.h5', 
                                monitor='val_loss',
                                verbose=1, 
                                save_best_only=True, 
                                save_weights_only=False, 
                                mode='min', 
                                period=1)
callback = [save_best, early_stop]



### compile 
model.compile(loss='categorical_crossentropy',
          optimizer='adam',
         metrics=['acc'])

### length of data
steps_per_epoch = np.floor(gen_train.get_length_data()*(1-val_size)) / batch_size 
val_setps_per_epoch = np.floor(gen_train.get_length_data() * val_size) / batch_size 

### fit
history = model.fit_generator(train_gen,
                    validation_data = val_gen,
                    steps_per_epoch = steps_per_epoch, 
                    validation_steps = val_setps_per_epoch,
                    epochs = 25,
                    callbacks = callback,
                    use_multiprocessing = False)

"""
            
#TODO FIX
# save as new model
folder_list = os.listdir('Models')
model_number_list = []
for item in folder_list:
    if item.spilt('.')[-1] == 'h5':
        try:
            name = item.split('.')[0]
            number = int(name.split('_')[-1])
        except:
            number = 99

        model_number_list.append(number)
        
prev_max_plus_one = np.amax(model_number_list) +1 

model_name = 'model_'+ str(prev_max_plus_one)
"""

#'model_summary' : model.summary(),
meta_data = {'model_name' : new_model_name,
                'batch_size' : batch_size,
                'train_path' : train_path,   
                'model_classes': gen_train.get_classes(),
                'model_augmentations' : gen_train.get_aug(),
                'model_history' :  history.history,
                'model_input_shape' : X_shape,
}
np.save(save_model_path +'meta_data_'+ new_model_name, meta_data)

#model.save(save_model_path + 'test_' + new_model_name)
