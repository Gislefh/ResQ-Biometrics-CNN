import sys
sys.path.insert(0, "classes")

import numpy as np
import matplotlib.pyplot as plt

from class_model import SecModel
from class_generator import Generator
from class_utils import get_vgg16_from_keras, get_vgg_w_imnet, add_classes_to_model

import keras
from keras.models import load_model
import h5py
import os

## paths
train_path = 'C:\\Users\\47450\\Documents\\ResQ Biometrics\\Data sets\\face-expression-recognition-dataset\\images\\train'
new_model_name = 'model_test_tensorboard.h5'
save_model_path = 'Models\\'


## consts
N_channels = 1
N_images_per_class = 4000
batch_size = 32
image_shape = (80, 80)
N_classes = 3
X_shape = (batch_size, image_shape[0], image_shape[1], N_channels)
Y_shape = (batch_size, N_classes)
val_size = 0.3


### generator
gen_train = Generator(train_path, X_shape, Y_shape, N_classes, N_channels, batch_size, class_list=['happy', 'neutral', 'angry'], train_val_split=val_size, N_images_per_class=N_images_per_class)
gen_train.add_rotate(max_abs_angle_deg=20)
gen_train.add_gamma_transform(0.5,1.5)
gen_train.add_flip()
gen_train.add_shift(0.1)
#gen_train.add_zoom(zoom_range= [0.2,2])

train_gen = gen_train.flow_from_dir(set = 'train')
val_gen = gen_train.flow_from_dir(set = 'val', augment_validation = True)



### -- get new model
m = SecModel(N_classes)
model = m.random_CNN(input_shape = (image_shape[0], image_shape[1], N_channels))
model.summary()

### -- vgg16 + empty
#model = get_vgg_w_imnet((image_shape[0], image_shape[1], N_channels), N_classes)

### -- fresh vgg
#model = get_vgg16_from_keras(input_shape, N_classes)


### --- load model
#model = load_model('Models\\model_10.h5')

## use pretrained model
#model = add_classes_to_model('Models\\model_9.h5', 3, 10)

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


tensorboard = keras.callbacks.TensorBoard(log_dir='C:\\Users\\47450\\Documents\\ResQ Biometrics\\ResQ-Biometrics-CNN\\Models\\Tensorboard', 
                                            histogram_freq=0, 
                                            batch_size=batch_size, 
                                            write_graph=True, 
                                            write_grads=False, 
                                            write_images=True, 
                                            embeddings_freq=0, 
                                            embeddings_layer_names=None, 
                                            embeddings_metadata=None, 
                                            embeddings_data=None, 
                                            update_freq='epoch')

callback = [tensorboard]

model.compile(loss='categorical_crossentropy',
          optimizer='adam',
         metrics=['acc'])

steps_per_epoch = np.floor(gen_train.get_length_data()*(1-val_size)) / batch_size 
val_setps_per_epoch = np.floor(gen_train.get_length_data() * val_size) / batch_size 

history = model.fit_generator(train_gen,
                    validation_data = val_gen,
                    steps_per_epoch = steps_per_epoch, 
                    validation_steps = val_setps_per_epoch,
                    epochs = 20,
                    callbacks = callback,
                    use_multiprocessing = False)


#                'model_summary' : model.summary(),
meta_data = {'model_name' : new_model_name,
                'batch_size' : batch_size,
                'train_path' : train_path,
                'model_classes': gen_train.get_classes(),
                'model_augmentations' : gen_train.get_aug(),
                'model_history' :  history.history,
                'model_input_shape' : X_shape,
}
np.save(save_model_path +'meta_data_'+ new_model_name, meta_data)

#model.save(save_model_path + _new_model_name + '.h5')