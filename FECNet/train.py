from generator import TripletGenerator
from model import faceNet_inceptionv3_model, test_siam_model, FECNet_inceptionv3_dense_model
from utils import distances
from custom_loss import TripletLoss
from custom_metrics import CustomMetrics
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
import keras
import os


data_path = 'C:\\Users\\47450\\Documents\\ResQ Biometrics\\Data sets\\FEC_dataset\\images\\two-class_triplets'
save_model_path = 'C:\\Users\\47450\\Documents\\ResQ Biometrics\\ResQ-Biometrics-CNN\\FECNet\\Models\\'
new_model_name = 'FECNet_dense_1.h5'
if new_model_name in os.listdir(save_model_path):
    print('--FROM SELF--: Model name exists. Change the model name')
    exit()

out_shape = (224, 224, 3)
delta_trip_loss = 0.1
embedding_size = 16 # faceNet uses 128, FECNet uses 16.
batch_size = 16
N_data_samples = None


# Data Generator 
trip_gen = TripletGenerator(data_path, out_shape = out_shape, batch_size=batch_size, augment=False, data = N_data_samples, train_val_split=0.05)
gen = trip_gen.flow_from_dir(set = 'train')
data_len = trip_gen.get_data_len(set = 'train')

val_gen = trip_gen.flow_from_dir(set = 'val')
val_data_len = trip_gen.get_data_len(set = 'val')


# Model
#model = FECNet_inceptionv3_model(input_shape = out_shape, embedding_size = embedding_size)
model = FECNet_inceptionv3_dense_model(input_shape=out_shape, embedding_size=embedding_size)
model.summary()


# Loss and metrics
L = TripletLoss(delta=delta_trip_loss, embedding_size=embedding_size)
M = CustomMetrics(embedding_size=embedding_size)

model.compile(loss=L.trip_loss,
              optimizer='adam')
              #metrics = [M.triplet_accuracy])

# callbacks


save_best = keras.callbacks.ModelCheckpoint(save_model_path + new_model_name,
                                            monitor='val_loss',
                                            verbose=1,
                                            save_best_only=True,
                                            save_weights_only=True,
                                            mode='min',
                                            period=1)

call = [save_best]
history = model.fit_generator(gen, steps_per_epoch=data_len/batch_size, 
                            validation_data=val_gen, validation_steps=val_data_len/batch_size,
                            epochs=200, shuffle=False, callbacks=call)



