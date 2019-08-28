import keras
import tensorflow
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


class Model:


	def __init__(self, N_classes):
		self.N_classes = N_classes



	def dynamic_model(input_shape, kernel_sizes, dropout, ):


	def compile_model(self, loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc']):

		
		self.model.compile(loss = loss,
				optimizer = optimizer,
				metrics = metrics)


		return self.model

	def random_CNN(self, input_shape):
		M = Sequential()
		M.add(Conv2D(32, (3, 3), activation='relu', input_shape = input_shape))
		M.add(Conv2D(32, (3, 3), activation='relu'))
		M.add(MaxPooling2D(pool_size=(2, 2)))
		M.add(Dropout(0.1))

		M.add(Conv2D(64, (3, 3), activation='relu'))
		M.add(Conv2D(64, (3, 3), activation='relu'))
		M.add(MaxPooling2D(pool_size=(2, 2)))
		M.add(Dropout(0.1))

		M.add(Conv2D(128, (3, 3), activation='relu'))
		M.add(Conv2D(128, (3, 3), activation='relu'))
		M.add(MaxPooling2D(pool_size=(2, 2)))
		M.add(Dropout(0.1))

		M.add(Conv2D(255, (3, 3), activation='relu'))
		M.add(Conv2D(255, (3, 3), activation='relu'))
		M.add(MaxPooling2D(pool_size=(2, 2)))
		M.add(Dropout(0.1))

		M.add(Conv2D(512, (3, 3), activation='relu'))
		M.add(Conv2D(512, (3, 3), activation='relu'))
		M.add(MaxPooling2D(pool_size=(2, 2)))
		M.add(Dropout(0.1))

		M.add(Flatten())
		M.add(Dense(512, activation='relu'))
		M.add(Dropout(0.1))
		M.add(Dense(512, activation='relu'))
		M.add(Dropout(0.1))
		M.add(Dense(self.N_classes, activation='softmax'))

		return M



