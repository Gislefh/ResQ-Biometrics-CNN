import sys
sys.path.insert(0, "classes")

import numpy as np
import matplotlib.pyplot as plt

from class_model import SecModel
from class_generator import Generator
from class_utils import get_vgg16_from_keras

import keras
from keras.models import load_model
import h5py


train_path = 'C:\\Users\\47450\\Documents\\ResQ Biometrics\\Data sets\\face-expression-recognition-dataset\\images\\train'
test_path = 'C:\\Users\\47450\\Documents\\ResQ Biometrics\\Data sets\\face-expression-recognition-dataset\\images\\validation'

N_channels = 1
batch_size = 8
image_shape = (70, 70)
N_classes = 7
X_shape = (batch_size, image_shape[0], image_shape[1], N_channels)
Y_shape = (batch_size, N_classes)
val_size = 0.3

gen_train = Generator(train_path, X_shape, Y_shape, N_classes, N_channels, batch_size, train_val_split = val_size)
gen_train.add_rotate(max_abs_angle_deg=20)
gen_train.add_gamma_transform(0.5,1.5)
gen_train.add_flip()
gen_train.add_shift(0.1)
#gen_train.add_zoom(zoom_range= [0.2,2])

train_gen = gen_train.flow_from_dir(set = 'train')
val_gen = gen_train.flow_from_dir(set = 'val', augment_validation = True)

gen_test = Generator(test_path, X_shape, Y_shape, N_classes, N_channels, batch_size)
test_gen = gen_test.flow_from_dir(set = 'test')


### get model
#m = SecModel(N_classes)
#model = m.random_CNN(input_shape = (image_shape[0], image_shape[1], N_channels))
#model.summary()


# load model
model = load_model('Models\\model2.h5')

## callbacks
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                min_delta=0, 
                                                patience=5, 
                                                verbose=0, 
                                                mode='auto', 
                                                baseline=None, 
                                                restore_best_weights=True)
callback = []
callback.append(early_stop)

model.compile(loss='categorical_crossentropy',
          optimizer='adam',
         metrics=['acc'])

steps_per_epoch = np.floor(gen_train.get_length_data()*(1-val_size)) / batch_size 
val_setps_per_epoch = np.floor(gen_train.get_length_data() * val_size) / batch_size 

history = model.fit_generator(train_gen,
                    validation_data = val_gen,
                    steps_per_epoch = steps_per_epoch, 
                    validation_steps = val_setps_per_epoch,
                    epochs = 15,
                    callbacks = callback,
                    use_multiprocessing = False)

print(history.history)
model_name = 'model2'  # <- change each training
save_model_path = 'Models\\'

meta_data = {'model_name' : model_name,
                'batch_size' : batch_size,
                'model_summary' : model.summary(),
                'model_classes': gen_test.get_classes(),
                'model_augmentations' : gen_test.get_aug(),
                'model_history' :  history,
                'model_input_shape' : X_shape,
}
np.save(save_model_path +'meta_data_'+ model_name, meta_data)

model.save(save_model_path + model_name + '.h5')