import numpy as np
import zipfile
import os
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from class_utils import one_hot

from pytictoc import TicToc
import dlib
import csv
import requests

#sys.path.insert(0, "C:\\Users\\47450\\Documents\\ResQ Biometrics\\ResQ-Biometrics-Model-1\\classes")
#from class_faceDetection import FaceDetection




class Generator:


	def __init__(self, path, X_shape, Y_shape, N_classes, N_channels, batch_size, train_val_split = 0.3, class_list = [], N_images_per_class = None):

		self.path = path 				#path to folder, str
		self.X_shape = X_shape			#shape of output, (samples, x, y, channel) or (samples, x, y, z, channels)
		self.Y_shape = Y_shape			#shape of ground truth, (samples, classes) for classification
		self.N_classes = N_classes		#number of classes, int
		self.N_channels = N_channels	#number of channels in the images, 1 is gray, 3 is color
		self.batch_size = batch_size	#number of samples in a batch
		self.image = None
		self.aug_method = []
		self.aug_args = []
		self.train_val_split = train_val_split 	#defaults to 0.3
		self.N_images_per_class = N_images_per_class
		if class_list:
			self.class_list = class_list
		else:
			self.class_list = []

	#returns the classes (folders) in the directory
	def get_classes(self):
		if self.class_list:
			return self.class_list
		else:
			class_list = []
			for folder in os.listdir(self.path):
				class_list.append(folder)
			return class_list

	# returns the chosen augmentations. Should be called after augmentations
	def get_aug(self):
		augs = []
		for aug in self.aug_method:
			augs.append(aug.__name__)
		return augs
	
	"""
	## works for all the data in the folders
	def get_length_data(self):
		N_data = 0
		for folder in os.listdir(self.path):
			if self.class_list:
				if folder in self.class_list:
					for file_ in os.listdir(self.path + '\\' +folder):
						N_data  = N_data + 1
			else:
				for file_ in os.listdir(self.path + '\\' +folder):
					N_data  = N_data + 1					
		return N_data
	"""
	def get_length_data(self):
		self.__from_dir(self.N_images_per_class)
		return len(self.train_set) + len(self.val_set)
	''' saves a list of derectories to images with the image class
	IN:
	N_images_per_class: how many images to get per class

	'''
	def __from_dir(self, N_images_per_class):

		image_list = []
		class_ = 0

		for folder in os.listdir(self.path):
			cnt_img_per_class = 0
			if self.class_list:
				if folder not in self.class_list:
					continue
				
			for image_ in os.listdir(self.path +'/' +folder):
				if N_images_per_class != None:
					if cnt_img_per_class > N_images_per_class:
						break ##TODO fix
					
				image_list.append([self.path + '/' + folder +'/' +image_, class_])
				cnt_img_per_class += 1
			
			class_ += 1

		## shuffle and save into test and train
		## TODO fix so that there are equal images for each class
		np.random.shuffle(image_list)
		image_list = np.array(image_list)

		self.train_set = image_list[0:int(image_list.shape[0] * (1-self.train_val_split)),  :]
		self.val_set = image_list[int(image_list.shape[0] * (1-self.train_val_split)):-1,  :]

	''' Creates a generator for either training set or validation set
	- IN:
	set: either val, train or test. str
	N_images_per_class: how many images to get per class
	train_val_split: how much of the data thats used as validaion
	'''

	def flow_from_dir(self, set = 'train', augment_validation = True):
		

		if set == 'test':
			self.train_val_split = 0

		# create sets
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
				self.image = cv2.imread( image_list[choice, 0])[:, :, 0:self.N_channels]
				label = int(image_list[choice, 1])

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

		
	### gets facial images from the web, displayes them and saves them in save_path in the chosen folder of the class. 
	def generator_from_web(self):

		#X_trip = np.zeros((self.X_shape[1] * 3, self.X_shape[2], self.X_shape[3]))


		url = ""
		#prev_url = ""

		with open(self.path, "r") as f:

			csv_reader = csv.reader(f, delimiter=",")
			counter = 0
			save_path = 'C:\\Users\\47450\\Documents\\ResQ Biometrics\\Data sets\\Data_set_from_web\\'
			combined_image = np.zeros((self.X_shape[1], self.X_shape[2]*3, self.X_shape[3]))


			#find the start point
			max_number = 0
			for folder in os.listdir(save_path):
				for file_ in  os.listdir(save_path +'\\'+ folder):
					image_name = file_.split('.')[0]
					if int(image_name) > max_number:
						max_number = int(image_name)


			for row in csv_reader:
				
				if counter < max_number:
					counter  += 3
					continue


				if row[15] !='ONE_CLASS_TRIPLET':
					continue

				imlist = []
				for row_inc in range(3):
					url = row[row_inc * 5]

					

					
					#if url != prev_url:
					decoded = cv2.imdecode(np.frombuffer(requests.get(url).content, np.uint8), -1)

					if len(np.shape(decoded)) != 3:
						counter += 1
						prev_url = url
						continue

					bb = []
					x1 = int(float(row[row_inc * 5 + 1]) * decoded.shape[1])
					x2 = int(float(row[row_inc * 5 + 2]) * decoded.shape[1])

					y1 = int(float(row[row_inc * 5 + 3]) * decoded.shape[0])
					y2 = int(float(row[row_inc * 5 + 4]) * decoded.shape[0])

					image = decoded
					image = image[y1:y2, x1:x2]
					image = self.__im_reshape(image.shape, image)
					imlist.append(image)


					counter += 1
					print(counter)

				
				
				for N, image_ in enumerate(imlist):
					combined_image[:, N * self.X_shape[2]: (N+1) *self.X_shape[2], :] = image_
				plt.imshow(combined_image /255)
				plt.show()

				chosen_class = input('press key to save as: 0-happy, 1-sad, 2-angry, 3-neutral, 4-surprise, 5-other, 6-tired/sleepy, 7 to skip images:   ')
				
				try:
					chosen_class = int(chosen_class)
				except:
					print('type in some of the options')
					continue
				
				
				folder_list = ['happy', 'sad', 'angry', 'neutral', 'surprise', 'other', 'tired_sleepy']

				if chosen_class == 7:
					print('skiped')
					continue

				elif len(folder_list) < chosen_class:
					print('choose from the list')

				
				else:
					path = save_path + folder_list[chosen_class] + '\\'
					for N, images in enumerate(imlist):
						cv2.imwrite(path + str(counter-2 + N) + '.jpg', images)
						#cv2.imwrite(path + str(counter -1) + '.jpg', imlist[1])
						#cv2.imwrite(path + str(counter) + '.jpg', imlist[2])
					print('Saved as:', folder_list[chosen_class])


	## yields a face from the google dataset	
	def face_from_web_gen(self, start_row = 0):
		prev_urls = []

		with open(self.path, "r") as f:

			csv_reader = csv.reader(f, delimiter=",")

			for row_nr, row in enumerate(csv_reader):

				# skip to start_row
				if row_nr < start_row:
					continue

				for row_inc in range(3):

					url = row[row_inc * 5]
					if url in prev_urls:
						continue

					prev_urls.append(url)

					image = cv2.imdecode(np.frombuffer(requests.get(url).content, np.uint8), -1)

					if len(np.shape(image)) != 3:
						#counter += 1
						continue

					bb = []
					x1 = int(float(row[row_inc * 5 + 1]) * image.shape[1])
					x2 = int(float(row[row_inc * 5 + 2]) * image.shape[1])

					y1 = int(float(row[row_inc * 5 + 3]) * image.shape[0])
					y2 = int(float(row[row_inc * 5 + 4]) * image.shape[0])

					image = image[y1:y2, x1:x2]
					image = self.__im_reshape(orig_shape = image.shape, image = image)

					yield image 


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


	# TODO fix this
	def generator_from_zip(self):
		pass
		"""
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
	

