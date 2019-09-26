import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from class_utils import one_hot

from pytictoc import TicToc
import csv


class GetDataset:


	def __init__(self, path, X_shape, Y_shape, N_classes, N_channels, train_val_split = 0.3, class_list = [], N_images_per_class = None):

		self.path = path 				#path to folder, str
		self.X_shape = X_shape			#shape of output, (samples, x, y, channel) or (samples, x, y, z, channels)
		self.Y_shape = Y_shape			#shape of ground truth, (samples, classes) for classification
		self.N_classes = N_classes		#number of classes, int
		self.N_channels = N_channels	#number of channels in the images, 1 is gray, 3 is color
		#self.batch_size = batch_size	#number of samples in a batch
		self.image = None
		self.aug_method = []
		self.aug_args = []
		self.train_val_split = train_val_split 	#defaults to 0.3
		self.N_images_per_class = N_images_per_class
		if class_list:
			self.class_list = class_list
		else:
			self.class_list = []

	def __from_dir(self, N_images_per_class):

		image_list = []
		class_ = 0

		tmp_val_set = []
		tmp_train_set = []

		for folder in os.listdir(self.path):
			cnt_img_per_class = 0
			if self.class_list:
				if folder not in self.class_list:
					continue
			
			## saving N_val first images as validation
			if N_images_per_class != None:
				N_val = int(N_images_per_class * self.train_val_split)
			else:
				N_val = int(len(os.listdir(self.path +'/' +folder)) * self.train_val_split)

			for image_ in os.listdir(self.path +'/' +folder):
				if N_images_per_class != None:
					if cnt_img_per_class > N_images_per_class:
						break ##TODO fix
				
				if cnt_img_per_class <= N_val:
					tmp_val_set.append([self.path + '/' + folder +'/' +image_, class_])
				else:
					tmp_train_set.append([self.path + '/' + folder +'/' +image_, class_])
				cnt_img_per_class += 1
			
			class_ += 1

		self.val_set = np.array(tmp_val_set)
		self.train_set = np.array(tmp_train_set)
		

	''' Creates a generator for either training set or validation set
	- IN:
	set: either val, train or test. str
	N_images_per_class: how many images to get per class
	train_val_split: how much of the data thats used as validaion
	'''
    def flow_from_dir(self, set = 'train', augment_validation = True):
		if set == 'test':
			self.train_val_split = 0

        self.__from_dir(self.N_images_per_class)

	    self.X = np.zeros(self.X_shape)
		self.Y = np.zeros(self.Y_shape)

		if set == 'train':
			tot_list = self.train_set
		elif set == 'val':
			tot_list = self.val_set
		elif set == 'test':
			tot_list = self.train_set
		else:
			print("select either: 'train', 'val' or 'test'")
			exit()
        

        ### place code here




        return (X, y)
    
    def flow_from_dir_old(self, set = 'train', augment_validation = True):
		

		if set == 'test':
			self.train_val_split = 0

		# create sets
		T = TicToc()
		self.__from_dir(self.N_images_per_class)
		
		self.X = np.zeros(self.X_shape)
		self.Y = np.zeros(self.Y_shape)

		if set == 'train':
			tot_list = self.train_set
		elif set == 'val':
			tot_list = self.val_set
		elif set == 'test':
			tot_list = self.train_set
		else:
			print("select either: 'train', 'val' or 'test'")
			exit()

		while True:

			image_list = tot_list.copy()

			for i in range(len(image_list)):
				##choose random image from list
				choice = np.random.choice(len(image_list[:, 0]))
				orig_ch = cv2.imread( image_list[choice, 0]).shape[-1]
				label = int(image_list[choice, 1])

				if (orig_ch == 3) and (self.N_channels == 1):
					im_tmp = cv2.imread( image_list[choice, 0])
					self.image = np.expand_dims(cv2.cvtColor(im_tmp, cv2.COLOR_BGR2GRAY), axis = -1)
				else:
					self.image = cv2.imread( image_list[choice, 0])[:, :, 0:self.N_channels]

			
				#normalize image to [0,1]
				self.image = np.clip(self.image / 255, 0, 1)
			

				### add augmentation	
				if (set == 'train') or (set == 'val' and augment_validation):
					for j, aug_method in enumerate(self.aug_method):
						if self.aug_args[j] == None:
							aug_method()
						else:
							aug_method(self.aug_args[j])
						
				## reshape image
				if self.image.shape != self.X[0].shape:
					self.X[i%self.batch_size] = self.__im_reshape(self.image.shape, self.image)
				else:
					self.X[i%self.batch_size] = self.image

				## one hot encode ground truth
				if self.class_list:
					self.Y[i%self.batch_size] = one_hot(label, len(self.class_list))
				else:
					self.Y[i%self.batch_size] = one_hot(label, self.N_classes)

				# delete this entry from the list
				image_list = np.delete(image_list, choice, 0)


				if i%self.batch_size == self.batch_size -1:
					yield(self.X, self.Y)

	''
	#### --- Image aug--

	#max_abs_angle_deg: maximum angle of rotation. positive scalar
	def add_rotate(self, max_abs_angle_deg = 10):
		self.aug_method.append(self.__rotate)
		self.aug_args.append(max_abs_angle_deg)
	
	#min: the lowest value 
	#max: the highest value
	#[min, max] shuld be in the range [0.3, 3], (isj)
	def add_gamma_transform(self, min, max):
		self.aug_method.append(self.__gamma_transfrom)
		self.aug_args.append([min, max])

	#max_shift: max normalized shift relative to the image shape. range [0,1]
	def add_shift(self, max_shift):
		self.aug_method.append(self.__shift)
		self.aug_args.append(max_shift)
	
	#flips image
	def add_flip(self):
		self.aug_method.append(self.__flip)
		self.aug_args.append(None)

	#zooms image in the range zoom_range
	def add_zoom(self, zoom_range):
		self.aug_method.append(self.__zoom)
		self.aug_args.append(zoom_range)		

	#-----
	def __gamma_transfrom(self, args):
		gamma = np.random.uniform(args[0], args[1])
		self.image = np.clip(np.power(self.image, gamma), 0, 1)

	def __rotate(self, max_angle):	
		angle = 2 * max_angle * np.random.rand() - max_angle 
		self.image =  ndimage.rotate(self.image, angle, reshape = False, order = 1)
		self.image = np.clip(self.image, 0, 1)
	
	def __shift(self, max_shift):
		shift_x = np.random.uniform(-self.image.shape[0], self.image.shape[0]) * max_shift
		shift_y = np.random.uniform(-self.image.shape[1], self.image.shape[1]) * max_shift
		self.image = ndimage.shift(self.image, (shift_x, shift_y, 0), order = 1)
		self.image = np.clip(self.image, 0, 1)

	def __flip(self):
		if np.random.rand() > 0.5:
			self.image = np.flip(self.image, axis = 1)
			self.image = np.clip(self.image, 0, 1)

	def __zoom(self, args):
		if np.random.rand() > 0.5:
			zoom_range_x = args[0] + (np.random.rand() * (args[1]- args[0]))
			zoom_range_y = 1
		else:
			zoom_range_x = 1
			zoom_range_y = args[0] + (np.random.rand() * (args[1]- args[0]))

		self.image = ndimage.zoom(self.image, (zoom_range_x, zoom_range_y, 1), order = 1)


	

	######## ---- utils ---

	def __im_reshape(self, orig_shape, image):
		
		factor_x = self.X_shape[1] / orig_shape[0]
		factor_y = self.X_shape[2] / orig_shape[1]

		return ndimage.zoom(image, (factor_x, factor_y, 1), order = 1)

