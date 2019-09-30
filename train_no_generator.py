import sys
sys.path.insert(0, "classes")
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from class_model import SecModel
from class_generator import Generator
from class_getDataset import GetDataset
from class_utils import get_vgg16_from_keras, get_vgg_w_imnet, add_classes_to_model, meta_data
from keras.preprocessing.image import ImageDataGenerator
import keras
from keras.models import load_model
import h5py
import tensorflow as tf
import os

## paths
train_path = 'C:\\Users\\47450\\Documents\\ResQ Biometrics\\Data sets\\ExpW\\train'
new_model_name = 'model_expw_preTr_vgg16_2_cont3.h5'
save_model_path = 'Models\\'

if new_model_name in os.listdir(save_model_path):
    print('Model name exists. Change the model name')
    exit()


## consts
N_channels = 3
N_images_per_class = 2000
image_shape = (100, 100)
N_classes = 7
X_shape = (image_shape[0], image_shape[1], N_channels)
val_size = 0.3
batch_size = 64


data_class = GetDataset(train_path, X_shape, N_classes, N_channels, N_images_per_class=N_images_per_class)
data_class.get_classes()
X, y = data_class.flow_from_dir()

X_train = X[0:int(np.shape(X)[0] *(1-val_size))]
X_val = X[0:int(np.shape(X)[0] *val_size)]
y_train = y[0:int(np.shape(X)[0] *(1-val_size))]
y_val = y[0:int(np.shape(X)[0] *val_size)]

data_gen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)

### -- get new model
#m = SecModel(N_classes)
#model = m.random_CNN(input_shape = (image_shape[0], image_shape[1], N_channels))
#model.summary()

### -- vgg16 + empty
#model = get_vgg_w_imnet((image_shape[0], image_shape[1], N_channels), N_classes)

### -- fresh vgg
#model = get_vgg16_from_keras((image_shape[0], image_shape[1], N_channels), N_classes)
#model.save(save_model_path + new_model_name)
#model = keras.applications.vgg16.VGG16(include_top=False, weights= None, input_tensor=None, input_shape=(image_shape[0], image_shape[1], N_channels), pooling=None, classes=N_classes)
#exit()

### --- load model
model = load_model('Models\\model_expw_preTr_vgg16_2_cont2.h5')

## use pretrained model
#model = add_classes_to_model('Models\\model_9.h5', 3, 10)


model.summary()

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

tensorboard = keras.callbacks.TensorBoard(log_dir='C:\\Users\\47450\\Documents\\ResQ Biometrics\\ResQ-Biometrics-CNN\\Models\\Tensorboard\\' + tensorboard_name, 
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


#history = model.fit(x=X, y=y, batch_size=batch_size, epochs=200, verbose=1, callbacks=callback, validation_split=val_size, validation_data=None, shuffle=False, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False)
history = model.fit_generator(data_gen.flow(X_train, y=y_train, batch_size=64, shuffle=False), 
                    validation_data = data_gen.flow(X_val, y=y_val, batch_size=64, shuffle=False),
                    steps_per_epoch = np.shape(X_train)[0] / batch_size, 
                    validation_steps = np.shape(X_val)[0] / batch_size,
                    epochs = 90,
                    callbacks = callback,
                    use_multiprocessing = False)                               

meta_data_dict = {'model_name' : new_model_name,
                'train_size' : N_images_per_class*N_classes*(1-val_size),
                'val_size' : N_images_per_class*N_classes*(val_size),
                'batch_size' : batch_size,
                'train_path' : train_path,
                'model_classes': data_class.get_classes(),
                'Images per class' : N_images_per_class,
                'model_input_shape' : X_shape,
                'N_epochs' : len(history.history['val_loss']),
                'best val loss' : min(history.history['val_loss']),
                'model_used' : 'vgg16 with imageNet'
}
meta_data(new_model_name, meta_data_dict)