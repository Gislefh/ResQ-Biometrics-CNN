import numpy as np
import zipfile
import os
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage, math
from class_utils import one_hot

from tqdm import tqdm
import dlib
import csv
import requests

from random import shuffle


class Generator:

    def __init__(self, path, X_shape, Y_shape, N_classes, N_channels, batch_size, train_val_split=0.3, class_list=[],
                 N_images_per_class=None):

        self.path = path  # path to folder, str
        self.X_shape = X_shape  # shape of output, (samples, x, y, channel) or (samples, x, y, z, channels)
        self.Y_shape = Y_shape  # shape of ground truth, (samples, classes) for classification
        self.N_classes = N_classes  # number of classes, int
        self.N_channels = N_channels  # number of channels in the images, 1 is gray, 3 is color
        self.batch_size = batch_size  # number of samples in a batch
        self.image = None
        self.aug_method = []
        self.aug_args = []
        self.train_val_split = train_val_split  # defaults to 0.3
        self.N_images_per_class = N_images_per_class
        self.class_list = (class_list if class_list else [])  # En sneaky Ternery :D

    # returns the classes (folders) in the directory
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

    # returns the number of images in the chosen classes. Called after the 'flow' function
    def get_length_data(self):
        self.__from_dir(self.N_images_per_class)
        return len(self.train_set) + len(self.val_set)

    ''' saves a list of derectories to images with the image class
	IN:
	N_images_per_class: how many images to get per class

	'''

    def __from_dir(self, N_images_per_class):
        class_ = 0

        tmp_val_set = []
        tmp_train_set = []

        for folder in os.listdir(self.path):
            if (folder.split('.')[-1] == "txt"):
                continue

            cnt_img_per_class = 0
            if self.class_list:
                if folder not in self.class_list:
                    continue

            ## saving N_val first images as validation
            if N_images_per_class != None:
                N_val = int(N_images_per_class * self.train_val_split)
            else:
                N_val = int(len(os.listdir(self.path + '/' + folder)) * self.train_val_split)

            for image_ in os.listdir(self.path + '/' + folder):
                if N_images_per_class != None:
                    if cnt_img_per_class > N_images_per_class:
                        break  ##TODO fix

                if cnt_img_per_class <= N_val:
                    tmp_val_set.append([self.path + '/' + folder + '/' + image_, class_])
                else:
                    tmp_train_set.append([self.path + '/' + folder + '/' + image_, class_])
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

    def flow_from_dir(self, set='train', augment_validation=True, crop=False):

        if set == 'test':
            self.train_val_split = 0

        # create sets
        self.__from_dir(self.N_images_per_class)

        self.X = np.zeros(self.X_shape, dtype=np.float32)
        self.Y = np.zeros(self.Y_shape, dtype=np.float32)

        if set == 'train':
            tot_list = self.train_set
        elif set == 'val':
            tot_list = self.val_set
        elif set == 'test':
            tot_list = self.train_set
        else:
            print("select either: 'train', 'val' or 'test'")
            exit()

        choice_list = list(range(len(tot_list)))

        while True:

            image_list = tot_list.copy()
            shuffle(choice_list)

            for i in range(len(image_list)):

                ##choose random image from list
                # choice = np.random.choice(len(image_list[:, 0]))
                choice = choice_list[i]
                orig_ch = cv2.imread(image_list[choice, 0]).shape[-1]
                label = int(image_list[choice, 1])

                if (orig_ch == 3) and (self.N_channels == 1):
                    im_tmp = cv2.imread(image_list[choice, 0])
                    self.image = np.expand_dims(cv2.cvtColor(im_tmp, cv2.COLOR_BGR2GRAY), axis=-1)
                else:
                    self.image = cv2.imread(image_list[choice, 0])[:, :, 0:self.N_channels]

                    # BGR to RGB
                    if '.jpg' in tot_list[choice, 0]:
                        # quickFix - should be improved
                        try:
                            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                        except:
                            print('Error when converting from BGR to RGB for image at location:', tot_list[choice, 0])
                if crop:
                    self.__crop()
                self.image = self.__im_reshape(self.image.shape, self.image)

                # normalize image to [0,1]
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
                    self.X[i % self.batch_size] = self.__im_reshape(self.image.shape, self.image)
                else:
                    self.X[i % self.batch_size] = self.image

                ## one hot encode ground truth
                if self.class_list:
                    self.Y[i % self.batch_size] = one_hot(label, len(self.class_list))
                else:
                    self.Y[i % self.batch_size] = one_hot(label, self.N_classes)

                # delete this entry from the list - nope not anymore
                # image_list = np.delete(image_list, choice, 0)

                if i % self.batch_size == self.batch_size - 1:
                    yield (self.X, self.Y)

    def flow_from_mem(self, set='train', augment_validation=True, fast_aug=True):
        if set == 'test':
            self.train_val_split = 0

        # create sets
        self.__from_dir(self.N_images_per_class)

        if set == 'train':
            tot_list = self.train_set
        elif set == 'val':
            tot_list = self.val_set
        elif set == 'test':
            tot_list = self.train_set
        else:
            print("select either: 'train', 'val' or 'test'")
            exit()

        self.X = np.zeros((len(tot_list), self.X_shape[1], self.X_shape[2], self.X_shape[3]), dtype=np.uint8)
        self.Y = np.zeros((len(tot_list), self.Y_shape[1]), dtype=np.uint8)

        image_list = tot_list.copy()
        choise_list = list(range(len(image_list)))

        for i in tqdm(range(len(image_list))):

            ##choose random image from list
            # choice = np.random.choice(len(image_list[:, 0]))
            choice = choise_list[i]
            orig_ch = cv2.imread(image_list[choice, 0]).shape[-1]
            label = int(image_list[choice, 1])

            if (orig_ch == 3) and (self.N_channels == 1):
                im_tmp = cv2.imread(image_list[choice, 0])
                self.image = np.expand_dims(cv2.cvtColor(im_tmp, cv2.COLOR_BGR2GRAY), axis=-1)
            else:
                self.image = cv2.imread(image_list[choice, 0])[:, :, 0:self.N_channels]

                # BGR to RGB
                if '.jpg' in tot_list[choice, 0]:
                    # quickFix - should be improved
                    try:
                        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                    except:
                        print('Error when converting from BGR to RGB for image at location:', tot_list[choice, 0])

            ## reshape image
            if self.image.shape != self.X[0].shape:
                self.X[i] = self.__im_reshape(self.image.shape, self.image)
            else:
                self.X[i] = self.image

            ## one hot encode ground truth
            if self.class_list:
                self.Y[i] = one_hot(label, len(self.class_list))
            else:
                self.Y[i] = one_hot(label, self.N_classes)

        # Fast augmentations - batch-wise
        if fast_aug:
            for i in range(len(self.aug_method)):
                if self.aug_method[i] == self.__flip:
                    self.aug_method[i] = self.__BW_flip

                if self.aug_method[i] == self.__gamma_transfrom:
                    self.aug_method[i] = self.__BW_gamma_transfrom

                if self.aug_method[i] == self.__rotate:
                    self.aug_method[i] = self.__BW_rotate

                if self.aug_method[i] == self.__shift:
                    self.aug_method[i] = self.__BW_shift

                if self.aug_method[i] == self.__zoom:
                    self.aug_method[i] = self.__BW_zoom

        self.X_out = np.zeros(self.X_shape, dtype=np.float32)
        self.Y_out = np.zeros(self.Y_shape, dtype=np.float32)

        N_batches = int(np.floor(len(image_list) / (self.batch_size)))

        while True:
            shuffle(choise_list)
            for i in range(N_batches):
                self.X_out = self.X[choise_list[i * self.batch_size: (1 + i) * self.batch_size]]
                self.Y_out = self.Y[choise_list[i * self.batch_size: (1 + i) * self.batch_size]]

                # normalize image to [0,1]
                self.X_out = np.clip(self.X_out / 255, 0, 1)

                ### add augmentation
                if (set == 'train') or (set == 'val' and augment_validation):
                    for j, aug_method in enumerate(self.aug_method):
                        if self.aug_args[j] == None:
                            aug_method()
                        else:
                            aug_method(self.aug_args[j])

                yield self.X_out, self.Y_out

    ### gets facial images from the web, displayes them and saves them in save_path in the chosen folder of the class.
    def generator_from_web(self):

        # X_trip = np.zeros((self.X_shape[1] * 3, self.X_shape[2], self.X_shape[3]))

        url = ""
        # prev_url = ""

        with open(self.path, "r") as f:

            csv_reader = csv.reader(f, delimiter=",")
            counter = 0
            save_path = 'C:\\Users\\47450\\Documents\\ResQ Biometrics\\Data sets\\Data_set_from_web\\'
            combined_image = np.zeros((self.X_shape[1], self.X_shape[2] * 3, self.X_shape[3]))

            # find the start point
            max_number = 0
            for folder in os.listdir(save_path):
                for file_ in os.listdir(save_path + '\\' + folder):
                    image_name = file_.split('.')[0]
                    if int(image_name) > max_number:
                        max_number = int(image_name)

            for row in csv_reader:

                if counter < max_number:
                    counter += 3
                    continue

                if row[15] != 'ONE_CLASS_TRIPLET':
                    continue

                imlist = []
                for row_inc in range(3):
                    url = row[row_inc * 5]

                    # if url != prev_url:
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
                    combined_image[:, N * self.X_shape[2]: (N + 1) * self.X_shape[2], :] = image_
                plt.imshow(combined_image / 255)
                plt.show()

                chosen_class = input(
                    'press key to save as: 0-happy, 1-sad, 2-angry, 3-neutral, 4-surprise, 5-other, 6-tired/sleepy, 7 to skip images:   ')

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
                        cv2.imwrite(path + str(counter - 2 + N) + '.jpg', images)
                    # cv2.imwrite(path + str(counter -1) + '.jpg', imlist[1])
                    # cv2.imwrite(path + str(counter) + '.jpg', imlist[2])
                    print('Saved as:', folder_list[chosen_class])

    ## yields a face from the google dataset
    def face_from_web_gen(self, start_row=0):
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
                        # counter += 1
                        continue

                    bb = []
                    x1 = int(float(row[row_inc * 5 + 1]) * image.shape[1])
                    x2 = int(float(row[row_inc * 5 + 2]) * image.shape[1])

                    y1 = int(float(row[row_inc * 5 + 3]) * image.shape[0])
                    y2 = int(float(row[row_inc * 5 + 4]) * image.shape[0])

                    image = image[y1:y2, x1:x2]
                    image = self.__im_reshape(orig_shape=image.shape, image=image)

                    yield image

                #### --- Image aug--

    # max_abs_angle_deg: maximum angle of rotation. positive scalar
    def add_rotate(self, max_abs_angle_deg=10):
        self.aug_method.append(self.__rotate)
        self.aug_args.append(max_abs_angle_deg)

    # min: the lowest value
    # max: the highest value
    # [min, max] shuld be in the range [0.3, 3], (isj)
    def add_gamma_transform(self, min, max):
        self.aug_method.append(self.__gamma_transfrom)
        self.aug_args.append([min, max])

    # max_shift: max normalized shift relative to the image shape. range [0,1]
    def add_shift(self, max_shift):
        self.aug_method.append(self.__shift)
        self.aug_args.append(max_shift)

    # flips image
    def add_flip(self):
        self.aug_method.append(self.__flip)
        self.aug_args.append(None)

    # zooms image in the range zoom_range
    def add_zoom(self, zoom_range):
        self.aug_method.append(self.__zoom)
        self.aug_args.append(zoom_range)

    def add_noise(self, noise=0.2):
        self.aug_method.append(self.__noise)
        self.aug_args.append(noise)

    def __noise(self, noise_sigma):
        h, w, d = self.image.shape
        self.image = self.image + np.float64(np.random.randn(h, w, d) * noise_sigma)

    def __crop(self):
        im_shape = np.shape(self.image)
        self.image = self.image[:, int(im_shape[1] / 4):int(3 * im_shape[1] / 4)]

    def __gamma_transfrom(self, args):
        gamma = np.random.uniform(args[0], args[1])
        self.image = np.clip(np.power(self.image, gamma), 0, 1)

    def __rotate(self, max_angle):
        angle = 2 * max_angle * np.random.rand() - max_angle
        self.image = ndimage.rotate(self.image, angle, reshape=False, order=1)
        self.image = np.clip(self.image, 0, 1)

    def __shift(self, max_shift):
        shift_x = np.random.uniform(-self.image.shape[0], self.image.shape[0]) * max_shift
        shift_y = np.random.uniform(-self.image.shape[1], self.image.shape[1]) * max_shift
        self.image = ndimage.shift(self.image, (shift_x, shift_y, 0), order=1)
        self.image = np.clip(self.image, 0, 1)

    def __flip(self):
        if np.random.rand() > 0.5:
            self.image = np.flip(self.image, axis=1)
            self.image = np.clip(self.image, 0, 1)

    def __zoom(self, args):
        if np.random.rand() > 0.5:
            zoom_range_x = args[0] + (np.random.rand() * (args[1] - args[0]))
            zoom_range_y = 1
        else:
            zoom_range_x = 1
            zoom_range_y = args[0] + (np.random.rand() * (args[1] - args[0]))

        self.image = ndimage.zoom(self.image, (zoom_range_x, zoom_range_y, 1), order=1)

    ## -- batch-wise aug --
    def __BW_gamma_transfrom(self, args):
        gamma = np.random.uniform(args[0], args[1])
        self.X_out = np.power(self.X_out, gamma)

    def __BW_rotate(self, max_angle):
        angle = 2 * max_angle * np.random.rand() - max_angle
        self.X_out = ndimage.rotate(self.X_out, angle, axes=[1, 2], reshape=False, order=1)

    def __BW_shift(self, max_shift):
        shift_x = np.random.uniform(-self.image.shape[0], self.image.shape[0]) * max_shift
        shift_y = np.random.uniform(-self.image.shape[1], self.image.shape[1]) * max_shift
        self.X_out = ndimage.shift(self.X_out, (0, shift_x, shift_y, 0), order=1)

    def __BW_flip(self):
        if np.random.rand() > 0.5:
            self.X_out = np.flip(self.X_out, axis=2)

    def __BW_zoom(self, args):  # dont use zoom
        if np.random.rand() > 0.5:
            zoom_range_x = args[0] + (np.random.rand() * (args[1] - args[0]))
            zoom_range_y = 1
        else:
            zoom_range_x = 1
            zoom_range_y = args[0] + (np.random.rand() * (args[1] - args[0]))

    # TODO fix so that self.X_out don't change shape
    # self.X_out = ndimage.zoom(self.X_out, (1, zoom_range_x, zoom_range_y, 1), order = 1)

    ######## ---- utils ---

    def __im_reshape(self, orig_shape, image):

        factor_x = self.X_shape[1] / orig_shape[0]
        factor_y = self.X_shape[2] / orig_shape[1]

        return ndimage.zoom(image, (factor_x, factor_y, 1), order=1)

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


if __name__ == '__main__':
    ## paths
    train_path = 'C:/Users/47450/Documents/ResQ Biometrics/Data sets/ExpW/train'

    ## consts
    N_channels = 3
    N_images_per_class = None
    batch_size = 64
    image_shape = (100, 100)
    N_classes = 6
    X_shape = (batch_size, image_shape[0], image_shape[1], N_channels)
    Y_shape = (batch_size, N_classes)
    val_size = 0.3

    ### generator
    gen_train = Generator(train_path, X_shape, Y_shape, N_classes, N_channels, batch_size,
                          train_val_split=val_size, N_images_per_class=N_images_per_class,
                          class_list=['angry', 'disgust', 'happy', 'neutral', 'sad', 'surprise'])

    # gen_train.add_rotate(max_abs_angle_deg=20)
    # gen_train.add_gamma_transform(0.5,1.5)
    # gen_train.add_flip()
    # gen_train.add_shift(0.1)

    train_gen = gen_train.flow_from_mem(set='train', fast_aug=True)
    train_gen_dir = gen_train.flow_from_dir(set='train')
    val_gen = gen_train.flow_from_mem(set='val', augment_validation=True, fast_aug=True)

    T = TicToc()
    T.tic()
    for i, (x, y) in enumerate(train_gen):

        print(i)
        if i > 20:
            break
    T.toc()
