import numpy as np
import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input

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

def get_vgg_w_imnet(input_shape, N_classes):

	init_model = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=input_shape, pooling='max', classes=None)

	x = Dense(255, activation='relu')(init_model.output)
	x = Dropout(0.1)(x)
	x = Dense(255, activation='relu')(x)
	x = Dropout(0.1)(x)
	preds = Dense(N_classes, activation='softmax')(x)
	model = Model(init_model.input, preds)
	

	## freeze all but the last 14 layers. last 6 conv2d and the dense layers
	for layer in model.layers[:-15]:
		layer.trainable = False


	return model


def add_classes_to_model(model_path, N_classes, freeze_N_layers = None):
	
	

	model = load_model(model_path)
	model.pop()
	model.add(Dense(N_classes, activation='softmax', name = 'pred_layer'))
	
	if freeze_N_layers != None:	
		for layer in model.layers[:-(freeze_N_layers+1)]:
			layer.trainable = False

	return model


def meta_data(model_name, data, path_to_folder = 'C:\\Users\\47450\\Documents\\ResQ Biometrics\\ResQ-Biometrics-CNN\Models\\Tensorboard\\metadata_models\\'):
	
	file1 = open(path_to_folder + model_name + '.txt', 'w') 
	file1.write('Model name: ' + model_name + '\n') 
	for key in data.keys():
		file1.write(key + ' : ' + str(data[key]) + '\n')
	file1.close()
  

def test_model():
	myInput = Input(shape=(96, 96, 3))

	x = Conv2D(32, (4, 4), activation='relu', input_shape=(96, 96, 3))(myInput)
	x = Conv2D(32, (4, 4), activation='relu')(x)
	M = Model(inputs=myInput, outputs=x)
	return M