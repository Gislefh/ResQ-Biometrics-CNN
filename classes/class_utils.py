import numpy as np
import keras


"""
--- From numerical value to one hot encoded ---
IN
	value: numerical value of class. int
	N_classes: number of classes. int

OUT
	one_hot: one hot encoded 1d-array
"""
def one_hot(value, N_classes):
	if N_classes < value:
		raise Exception("Can't one hot encode value outside the range")

	one_hot = np.zeros((N_classes))
	one_hot[value] = 1
	return one_hot


def get_vgg16_from_keras(input_shape, N_classes):
		if input_shape == None:
			inc_top = True
			inp_shape = None

		else:
			inc_top = False
			inp_shape = input_shape

		model = keras.applications.vgg16.VGG16(include_top=inc_top, weights= None, input_tensor=None, input_shape=inp_shape, pooling=None, classes=N_classes)
		return model

