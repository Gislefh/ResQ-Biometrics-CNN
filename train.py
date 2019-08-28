import sys
sys.path.insert(0, "classes")

import numpy as np
import matplotlib.pyplot as plt
from class_model import Model
from class_generator import Generator
import keras


path = '/home/gisleh/Downloads/face-expression-recognition-dataset/images/train'
N_channels = 1
batch_size = 10
image_shape = (170, 170)
N_classes = 7
X_shape = (batch_size, image_shape[0], image_shape[1], N_channels)
Y_shape = (batch_size, N_classes)


gen = Generator(path, X_shape, Y_shape, N_classes, N_channels, batch_size)

batch_gen = gen.generator_from_dir()


m = Model(N_classes)
model = m.random_CNN(input_shape = (image_shape[0], image_shape[1], 1))
model.summary()

model.compile(loss='categorical_crossentropy',
          optimizer='adam',
         metrics=['acc'])


model.fit_generator(batch_gen, steps_per_epoch = 1000, epochs = 20)