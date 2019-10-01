import sys
sys.path.insert(0, "classes")
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from class_model import SecModel
from class_generator import Generator
from class_getDataset import GetDataset
from class_customCallback import CustomCallback
from class_utils import get_vgg16_from_keras, get_vgg_w_imnet, add_classes_to_model, meta_data
from keras.preprocessing.image import ImageDataGenerator
import keras
from keras.models import load_model
import h5py
import tensorflow as tf
import os

## paths
train_path = 'C:\\Users\\47450\\Documents\\ResQ Biometrics\\Data sets\\ExpW\\train'

new_model_name = 'model_expw_preTr_vgg16_4.h5'
save_model_path = 'Models\\'

if new_model_name in os.listdir(save_model_path):
    print('Model name exists. Change the model name')
    exit()


## consts
N_channels = 3
N_images_per_class = 3000
image_shape = (200, 200)
N_classes = 7
X_shape = (image_shape[0], image_shape[1], N_channels)
val_size = 0.3
batch_size = 64


data_class = GetDataset(train_path, X_shape, N_classes, N_channels, N_images_per_class=N_images_per_class)
data_class.get_classes()
X, y = data_class.flow_from_dir()

X_train = X[0:int(np.shape(X)[0] *(1-val_size))]
X_val = X[int(np.shape(X)[0] *(1-val_size)):-1]
y_train = y[0:int(np.shape(X)[0] *(1-val_size))]
y_val = y[int(np.shape(X)[0] *(1-val_size)):-1]

data_gen_train = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, zoom_range=0.1)
data_gen_val = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, zoom_range=0.1)

### -- get new model
#m = SecModel(N_classes)
#model = m.random_CNN(input_shape = (image_shape[0], image_shape[1], N_channels))
#model.summary()

### -- vgg16 w imagenet + empty dense layers
model = get_vgg_w_imnet((image_shape[0], image_shape[1], N_channels), N_classes)

### -- fresh vgg
#model = get_vgg16_from_keras((image_shape[0], image_shape[1], N_channels), N_classes)
#model.save(save_model_path + new_model_name)
#model = keras.applications.vgg16.VGG16(include_top=False, weights= None, input_tensor=None, input_shape=(image_shape[0], image_shape[1], N_channels), pooling=None, classes=N_classes)
#exit()

### --- load model
#model = load_model('Models\\model_expw_preTr_vgg16_2_cont2.h5')

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


#tensorboard_name = datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_name = new_model_name.split('.')[0]
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


history = model.fit_generator(data_gen_train.flow(X_train, y=y_train, batch_size=batch_size, shuffle=False), 
                    validation_data = data_gen_val.flow(X_val, y=y_val, batch_size=batch_size, shuffle=False),
                    steps_per_epoch = np.shape(X_train)[0] / batch_size, 
                    validation_steps = np.shape(X_val)[0] / batch_size,
                    epochs = 100,
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