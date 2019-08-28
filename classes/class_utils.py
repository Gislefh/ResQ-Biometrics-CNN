import numpy as np



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

