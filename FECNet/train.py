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
out_shape = (224, 224, 3)
delta_trip_loss = 0.1
embedding_size = 16 # faceNet uses 128, FECNet uses 16.
batch_size = 64



# Data Generator 
trip_gen = TripletGenerator(path, out_shape = out_shape, batch_size=batch_size)
gen = trip_gen.flow_from_dir()
data_len = trip_gen.get_data_len()

# Model
model = faceNet_inceptionv3_model(input_shape = out_shape, embedding_size = embedding_size)
#model = test_siam_model(input_shape=out_shape, embedding_size=embedding_size)
model.summary()


# Loss
L = TripletLoss(delta=delta_trip_loss, embedding_size=embedding_size)


model.compile(loss=L.trip_loss,
              optimizer='adam')

# callbacks
save_model_path = 'C:\\Users\\47450\\Documents\\ResQ Biometrics\\ResQ-Biometrics-CNN\\FECNet\\Models\\'
new_model_name = 'FECNet_test1.h5'
save_best = keras.callbacks.ModelCheckpoint(save_model_path + new_model_name,
                                            monitor='loss',
                                            verbose=1,
                                            save_best_only=True,
                                            save_weights_only=True,
                                            mode='min',
                                            period=1)


history = model.fit_generator(gen, steps_per_epoch=data_len/batch_size, epochs=3, shuffle=False, callbacks=[save_best])



