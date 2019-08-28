import numpy as np
import zipfile
import os
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from class_utils import one_hot
class Generator:


	def __init__(self, path, X_shape, Y_shape, N_classes, N_channels, batch_size):

		self.path = path 				#path to folder, str
		self.X_shape = X_shape			#shape of output, (samples, x, y, channel) or (samples, x, y, z, channels)
		self.Y_shape = Y_shape			#shape of ground truth, (samples, classes) for classification
		self.N_classes = N_classes		#number of classes, int
		self.N_channels = N_channels	#number of channels in the images, 1 is gray, 3 is color
		self.batch_size = batch_size	#number of samples in a batch


	def __im_reshape(self, orig_shape, image):
		
		factor_x = self.X_shape[1] / orig_shape[0]
		factor_y = self.X_shape[2] / orig_shape[1]

		return ndimage.zoom(image, (factor_x, factor_y, 1), order = 3)







	def generator_from_dir(self):


		if self.N_classes != len(os.listdir(self.path)):
			print('the number of classes does not match the number of folders')
			return None

		self.X = np.zeros(self.X_shape)
		self.Y = np.zeros(self.Y_shape)


		while True:

			image_list = []

			for class_, folder in enumerate(os.listdir(self.path)):
				for image_ in os.listdir(self.path +'/' +folder):
					image_list.append([self.path + '/' + folder +'/' +image_, class_])
			image_list = np.array(image_list)

			for i in range(len(image_list)):
				choice = np.random.choice(len(image_list[:, 0]))
				image = cv2.imread( image_list[choice, 0])[:, :, 0:1]
				label = int(image_list[choice, 1])


				if image.shape != self.X[0].shape:
					self.X[i%self.batch_size] = self.__im_reshape(image.shape, image)
				else:
					self.X[i%self.batch_size] = image

				self.Y[i%self.batch_size] = one_hot(label, self.N_classes)

				if i%self.batch_size == self.batch_size -1:
					#return self.X, self.Y
					yield(self.X, self.Y)

		



	def image_aug(self):
		pass

	#def __flow(self):
	#	while True:
			
	#		yield (X, Y)

"""
	# TODO fix this
	def generator_from_zip(self):
		file_name = 'face-expression-recognition-dataset.zip'
		cnt = 0
		tot_cnt = 0
		N_images = batch_size
		N_channels = 1
		N_classes = 2
		im_shape_x = input_shape[0]
		im_shape_y = input_shape[1]



		#Images
		X = np.zeros((N_images, im_shape_x, im_shape_y, N_channels), dtype = np.uint8)

		# Ground truth. [angry, neutral]
		Y = np.zeros((N_images, N_classes), dtype = np.uint8) 



		with zipfile.ZipFile(file_name, 'r') as zip:
	
	
			while True:
				random.shuffle(zip.infolist())
		
				for entry in zip.infolist():


					with zip.open(entry) as file:


						if not 'jpg' in file.name:
							continue

						else:

							if cnt >= N_images:
								cnt = 0
							
								yield (X, Y)

							elif 'angry' in file.name:
								Y[cnt, :] = [1, 0] 
								img = Image.open(file)
							  
								#resize
								scale_fac = input_shape[0]/np.shape(img)[0]
								img_large = ndimage.zoom(img, scale_fac, order = 3)
								X[cnt, :, :, 0] = img_large
								cnt += 1
								tot_cnt += 1

							elif 'surprise' in file.name:
								Y[cnt, :] = [0, 1] 
								img = Image.open(file)
							  
								#resize
								scale_fac = input_shape[0]/np.shape(img)[0]              
								img_large = ndimage.zoom(img, scale_fac, order = 3)

								X[cnt, :, :, 0] = img_large

								cnt += 1
								tot_cnt += 1



							elif tot_cnt > 100:
								tot_cnt = 0
								break


							else: 
								continue
"""		

