import keras
import tensorflow
import numpy as np



class Model:


	def __init__(N_classes):
		self.N_classes = N_classes

	def vgg16_from_keras(self, input_shape):
		if input_shape == None:
			inc_top = True
			inp_shape = None

		else:
			inc_top == False
			inp_shape = input_shape

		self.model = keras.applications.vgg16.VGG16(include_top=inc_top, weights= None, input_tensor=None, input_shape=inp_shape, pooling=None, classes=self.N_classes)

	def compile_model(self, loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc']):

		
		self.model.compile(loss = loss,
				optimizer = optimizer,
				metrics = metrics)


		return self.model




