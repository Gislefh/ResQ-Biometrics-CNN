import tensorflow as tf 
from generator import TripletGenerator
from model import FECNet_inceptionv3_model
from utils import distances, pca_FECNet, eval_gen
import numpy as np
import matplotlib.pyplot as plt
from custom_loss import TripletLoss

path = 'C:\\Users\\47450\\Documents\\ResQ Biometrics\\Data sets\\FEC_dataset\\images\\two-class_triplets'
save_model_path = 'C:\\Users\\47450\\Documents\\ResQ Biometrics\\ResQ-Biometrics-CNN\\FECNet\\Models\\'
new_model_name = 'MNIST_digits_test.h5'
out_shape = (224, 224, 3)
embedding_size = 16 # faceNet uses 128, FECNet uses 16.
batch_size = 1
delta_trip_loss = 0.2

# Data Generator
trip_gen = TripletGenerator(path, out_shape = out_shape, batch_size=batch_size, augment=True, data=5000)
gen = trip_gen.flow_from_dir()
data_len = trip_gen.get_data_len()


# Model
model = FECNet_inceptionv3_model(input_shape = out_shape, embedding_size = embedding_size)
model.load_weights('Models\\FECNet_test1.h5')


eval_gen(gen, model, data_len, embedding_size)
exit()


pred = []
images = []
for i, (x,y) in enumerate(gen):
    if i >= data_len:
        break
    
    pred.append(model.predict(x, batch_size = batch_size, steps = 1))
    print('predicted on ' , i, ' tripets')

pca_FECNet(pred, embedding_size, I, N_comp = 2)







