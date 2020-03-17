from datetime import datetime

import cv2
import numpy as np
from class_generator import Generator
from RNN_model import RnnCnnModel

import keras
from keras.models import load_model
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

## paths
train_path = 'C:\\ML\\Dataset\\videos\\'
new_model_name = 'XceptionV3+RNN_videos_0'
save_model_path = 'C:\\ML\\Models\\CNN\\'

if new_model_name in os.listdir(save_model_path):
    print('Model name exists. Change the model name')
    # exit()

# TODO: Remove batch size
n_channels = 3
N_images_per_class = None
batch_size = 16
image_shape = (72 * 1, 64 * 1)
N_classes = 8
X_shape = (1, image_shape[0], image_shape[1], n_channels)
Y_shape = (1, N_classes)
val_size = 0.2

g = Generator(train_path,
              image_shape,
              X_shape,
              Y_shape,
              N_classes,
              val_size
              )

train_data = g.get_data("train", batch_size)
val_data = g.get_data("val", batch_size)
# model = load_model('C:\\ML\\Models\\CNN\\XceptionV3_videoframes_1.h5')
m = RnnCnnModel(N_classes)
model = m.rnnCnnModel(input_shape=(image_shape[0], image_shape[1], n_channels))
model.summary()

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
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

tensorboard_name = datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard = keras.callbacks.TensorBoard(log_dir='C:\\ML\\Models\\CNN\\new\\' + tensorboard_name,
                                          histogram_freq=0,
                                          batch_size=batch_size,
                                          write_graph=True,
                                          write_grads=True,
                                          write_images=True,
                                          embeddings_freq=0,
                                          embeddings_layer_names=None,
                                          embeddings_metadata=None,
                                          embeddings_data=None,
                                          update_freq='batch')

callback = [tensorboard, save_best, early_stopping]

steps_per_epoch = np.floor(g.get_data_length() * (1 - val_size)) / batch_size

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

# for (x, y) in train_data:
#     for i in range(len(x[0])):
#         cv2.imshow("", x[0, i])
#         cv2.waitKey()

history = model.fit_generator(train_data, validation_data=val_data, validation_steps=g.get_data_length("val"),
                              steps_per_epoch=steps_per_epoch,
                              epochs=100,
                              callbacks=callback,
                              verbose=1,
                              use_multiprocessing=False)

# TODO: Generator returner 5D, skal returne noe annet
#
