import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout


def one_hot(value, N_classes):
	output = np.zeros((N_classes))
	output[int(value)] = 1
	return output


def get_inception_w_imnet(input_shape, N_classes, freeze_layers=False):
	init_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet', input_shape=input_shape,
	                                               pooling='max')
	
	x = Dense(255, activation='relu')(init_model.output)
	x = Dropout(0.3)(x)
	preds = Dense(N_classes, activation='softmax')(x)
	model = Model(init_model.input, preds)
	
	## freeze all but the last n layers
	if freeze_layers:
		for layer in model.layers[:-68]:
			layer.trainable = False
	return model
