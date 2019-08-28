import sys
sys.path.insert(0, "classes")

import numpy as np
import matplotlib.pyplot as plt
from class_model import SecModel
from class_generator import Generator
from class_utils import get_vgg16_from_keras
import keras


path = 'C:\\Users\\47450\\Documents\\ResQ Biometrics\\Data sets\\training_set\\training_set'
N_channels = 3
batch_size = 10
image_shape = (130, 130)
N_classes = 2
X_shape = (batch_size, image_shape[0], image_shape[1], N_channels)
Y_shape = (batch_size, N_classes)


gen = Generator(path, X_shape, Y_shape, N_classes, N_channels, batch_size)

batch_gen = gen.generator_from_dir()


#m = SecModel(N_classes)
#model = m.random_CNN(input_shape = (image_shape[0], image_shape[1], 1))
#model.summary()

model = get_vgg16_from_keras(input_shape = (image_shape[0], image_shape[1], N_channels), N_classes = N_classes) 
model.summary()


model.compile(loss='categorical_crossentropy',
          optimizer='adam',
         metrics=['acc'])


steps_per_epoch = 8000/batch_size


model.fit_generator(batch_gen, steps_per_epoch = steps_per_epoch, epochs = 20)


""" --- test batch gen ----
for x,y in batch_gen:
    for i in range(np.shape(x)[0]):
        print(y[i], np.amax(x[i]))
        plt.imshow(x[i])
        plt.show()
    break

"""