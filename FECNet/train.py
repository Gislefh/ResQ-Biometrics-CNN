from generator import TripletGenerator
from model import faceNet_inceptionv3_model, test_siam_model
from custom_loss import TripletLoss
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

path = 'C:\\Users\\47450\\Documents\\ResQ Biometrics\\Data sets\\FEC_dataset\\images\\two-class_triplets'
out_shape = (1, 224, 224, 3)
delta_trip_loss = 0.1
embedding_size = 1 # faceNet uses 128, FECNet uses 16.


# Data Generator 
trip_gen = TripletGenerator(path, out_shape = out_shape)
gen = trip_gen.flow_from_dir()
data_len = trip_gen.get_data_len()


# Model
#model = faceNet_inceptionv3_model(input_shape = out_shape[1:], embedding_size = embedding_size)
model = test_siam_model(input_shape=out_shape[1:], embedding_size=embedding_size)
model.summary()


# Loss
L = TripletLoss(delta=delta_trip_loss, embedding_size=embedding_size)


model.compile(loss=L.trip_loss,
              optimizer='adam')

history = model.fit_generator(gen, steps_per_epoch=data_len, epochs=2, shuffle=False)


'''

for x,y in gen:
    print(len(x))
    print(np.shape(x[0]))
    print(y)
    
    pred = model.predict(x)
    loss = L.trip_loss(y,pred)
    from tensorflow.python.keras import backend as K
    loss_mean = K.mean(loss)
    sess = K.get_session()
    array = sess.run(loss)
    array_mean = sess.run(loss_mean)
    print(array, array_mean)
    
    exit()
'''

