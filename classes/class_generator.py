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







	def generator_from_dir(self, include_folder_list = []):


		#if self.N_classes != len(os.listdir(self.path)):
		#	print('the number of classes does not match the number of folders')
		#	return None

		self.X = np.zeros(self.X_shape)
		self.Y = np.zeros(self.Y_shape)


		while True:

			image_list = []
			class_ = 0
			for folder in os.listdir(self.path):
				if include_folder_list:
					if folder not in include_folder_list:
						continue
					
				for image_ in os.listdir(self.path +'/' +folder):
					image_list.append([self.path + '/' + folder +'/' +image_, class_])
				
				class_ += 1

			image_list = np.array(image_list)

			for i in range(len(image_list)):
				choice = np.random.choice(len(image_list[:, 0]))
				image = cv2.imread( image_list[choice, 0])[:, :, 0:self.N_channels]
				
				
				# normalize
				image = image / 255

				label = int(image_list[choice, 1])
				


				if image.shape != self.X[0].shape:
					self.X[i%self.batch_size] = self.__im_reshape(image.shape, image)
				else:
					self.X[i%self.batch_size] = image

				self.Y[i%self.batch_size] = one_hot(label, len(np.unique(image_list[:,1])))

				#image_list = np.delete(image_list, choice, 0)


				if i%self.batch_size == self.batch_size -1:
					#return self.X, self.Y
					yield(self.X, self.Y)

		

	def generator_from_web(self):

		#X_trip = np.zeros((self.X_shape[1] * 3, self.X_shape[2], self.X_shape[3]))


		url = ""
		prev_url = ""

		with open(self.path, "r") as f:

			csv_reader = csv.reader(f, delimiter=",")
			counter = 0
			for row in csv_reader:



				if row[15] !='ONE_CLASS_TRIPLET':
					continue

				
				for row_inc in range(3):
					url = row[row_inc * 5]

					

					
					if url != prev_url:
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


						
						image = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
						image = image[y1:y2, x1:x2]
						image = self.__im_reshape(image.shape, image)

						#detector = dlib.get_frontal_face_detector()
						#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
						#faces = detector(gray, 1)

						#print(faces)
						plt.figure(row_inc)
						plt.imshow(image)
						#plt.show()
						



						counter += 1
						prev_url = url

						print(counter)
						if counter > 100:
							exit()
				plt.show()				


			

	def image_aug(self):
		pass

	#def __flow(self):
	#	while True:
			
	#		yield (X, Y)


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
	

