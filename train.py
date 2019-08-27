import sys
sys.path.insert(0, "classes")

import numpy as np
import matplotlib.pyplot as plt
from class_model import Model
from class_generator import Generator


path = '/home/gisleh/Downloads/face-expression-recognition-dataset/images/train'
N_channels = 1
batch_size = 10
image_shape = (100, 100)
N_classes = 7
X_shape = (batch_size, image_shape[0], image_shape[1], N_channels)
Y_shape = (batch_size, N_classes)


gen = Generator(path, X_shape, Y_shape, N_classes, N_channels, batch_size)

X = gen.generator_from_dir()

for i in range(np.shape(X)[0]):
	plt.imshow(X[i, :, :, 0], cmap = 'gray')
	plt.show()
