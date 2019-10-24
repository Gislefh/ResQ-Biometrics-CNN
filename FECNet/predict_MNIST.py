import tensorflow as tf 
from generator import TripletTestGenerator
from model import test_siam_model
from utils import distances, pca_MNIST
import numpy as np
import matplotlib.pyplot as plt

save_model_path = 'C:\\Users\\47450\\Documents\\ResQ Biometrics\\ResQ-Biometrics-CNN\\FECNet\\Models\\'
new_model_name = 'MNIST_digits_test.h5'
out_shape = (28,28)
embedding_size = 16 # faceNet uses 128, FECNet uses 16.
batch_size = 8

batch_size = 1


G = TripletTestGenerator(out_shape, batch_size)
gen = G.flow()



#weights = tf.keras.models.load_weights('Models\\MNIST_digits_test.h5')

model = test_siam_model(input_shape=out_shape, embedding_size=embedding_size)
model.load_weights('Models\\MNIST_digits_test_only_weights.h5')


from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


x = []
y = []
for i in range(len(y_test)):
    y.append(y_test[i])
    if i%3 == 2:
        x.append([np.expand_dims(x_test[i-2], axis = 0), np.expand_dims(x_test[i-1], axis = 0), np.expand_dims(x_test[i], axis = 0)])
        

pred = []
for i, samp in enumerate(x):
    pred.append(model.predict(samp, batch_size = batch_size, steps = 1))

pca_MNIST(pred, embedding_size, y, N_comp=2)


exit()
    


cnt = 0
pred = []
for x,y in gen:
    pred.append(model.predict(x, batch_size = batch_size, steps = 1))
    cnt += 1
    if cnt > 200:
        pca(pred, embedding_size)
        break





