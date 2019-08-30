import keras
import tensorflow
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


class SecModel:


	def __init__(self, N_classes):
		self.N_classes = N_classes
		self.model = Sequential()
		self.set_filter_list = []

	#def set_filters(self, ):


	"""
	returns a squential model

	IN:
		input_shape: shape of the data coming in
		kernel_shape: int or tuple
		dropout: depth of network, int
		depth: 

	"""
	def dynamic_model(self, input_shape, kernel_sizes, depth, dropout = 0.1, activation_conv = 'relu', activation_out = 'softmax', max_pool_shape = (2,2)):

		self.model.add(Conv2D(2**5, (kernel_sizes, kernel_sizes), activation = 'relu', input_shape = input_shape))
		self.model.add(Conv2D(2**5, (kernel_sizes, kernel_sizes)))
		self.model.add(MaxPooling2D(pool_size=max_pool_shape))
		#Convolution
		for layer in range(len(depth - 1)):
			filters = 2 ** (layer+6)
			self.model.add(Conv2D(filters, (kernel_sizes, kernel_sizes)))
			self.model.add(Conv2D(filters, (kernel_sizes, kernel_sizes)))
			self.model.add(MaxPooling2D(pool_size=max_pool_shape))
			if dropout:
				self.model.add(Dropout(dropout))

		#Dense

		self.model.add(Dense(512, activation='relu'))
		self.model.add(Dropout(dropout))
		self.model.add(Dense(512, activation='relu'))
		self.model.add(Dropout(dropout))
		self.model.add(Dense(self.N_classes, activation='softmax'))

		return model




	def compile_model(self, loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc']):

		
		self.model.compile(loss = loss,
				optimizer = optimizer,
				metrics = metrics)


		return self.model

	def random_CNN(self, input_shape):
		M = Sequential()
		M.add(Conv2D(32, (5, 5), activation='relu', input_shape = input_shape))
		M.add(Conv2D(32, (5, 5), activation='relu'))
		M.add(MaxPooling2D(pool_size=(2, 2)))
		#M.add(Dropout(0.1))

		M.add(Conv2D(64, (5, 5), activation='relu'))
		M.add(Conv2D(64, (5, 5), activation='relu'))
		M.add(MaxPooling2D(pool_size=(2, 2)))
		#M.add(Dropout(0.1))

		M.add(Conv2D(128, (5, 5), activation='relu'))
		M.add(Conv2D(128, (5, 5), activation='relu'))
		M.add(MaxPooling2D(pool_size=(2, 2)))
		#M.add(Dropout(0.1))

		#M.add(Conv2D(255, (3, 3), activation='relu'))
		#M.add(Conv2D(255, (3, 3), activation='relu'))
		#M.add(MaxPooling2D(pool_size=(2, 2)))
		#M.add(Dropout(0.1))

		#M.add(Conv2D(512, (3, 3), activation='relu'))
		#M.add(Conv2D(512, (3, 3), activation='relu'))
		#M.add(MaxPooling2D(pool_size=(2, 2)))
		#M.add(Dropout(0.1))

		M.add(Flatten())
		M.add(Dense(512, activation='relu'))
		#M.add(Dropout(0.1))
		M.add(Dense(512, activation='relu'))
		#M.add(Dropout(0.1))
		M.add(Dense(self.N_classes, activation='softmax'))

		return M



