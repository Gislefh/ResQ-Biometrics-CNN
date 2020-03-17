import os
from random import shuffle
import cv2
import numpy as np
from scipy.ndimage import zoom
from sys import getsizeof


class Generator:
    def __init__(self, path, image_shape, x_shape, y_shape, n_classes, val_size):
        self.path = path
        self.image_x_shape = image_shape[1]
        self.image_y_shape = image_shape[0]
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.n_classes = n_classes
        self.validation_set = []
        self.train_set = []

        self.make_train_val_split(val_size)

    def make_train_val_split(self, val_percent=0.2):
        video_format = ".mp4"

        paths = []
        for r, d, f in os.walk(self.path):
            for file in f:
                if video_format in file:
                    paths.append(os.path.join(r, file))

        shuffle(paths)
        val_size = len(paths) * val_percent
        counter = 0

        for path in paths:
            # TODO: Might be able to improve speed of this
            if counter > val_size:
                self.train_set.append(path)
            else:
                self.validation_set.append(path)
            counter += 1

    def one_hot(self, value, N_classes):
        value -= 1
        if N_classes < value:
            raise Exception("-- FROM SELF -- Can't one hot encode value outside the range")

        one_hot = np.zeros((N_classes))
        one_hot[value] = 1
        return one_hot

    def video_path_to_images(self, path):
        cap = cv2.VideoCapture(path)

        frame_count, frame_height, frame_width = self.get_video_shape(cap)

        buffer = np.zeros((1, frame_count, self.image_y_shape, self.image_x_shape, 3), np.dtype('uint8'))

        im_shape = (frame_height, frame_width, 3)
        y_factor = self.image_y_shape / frame_height
        x_factor = self.image_x_shape / (frame_width / 2)

        index = 0
        cnt = 0
        ret = True
        while index < frame_count and ret:
            ret, image = cap.read()
            cnt += 1
            if cnt % 3 == 0:
                continue
            if image is not 0 and ret:
                image = image[:, int(im_shape[1] / 4):int(3 * im_shape[1] / 4)]
                image = zoom(image, (x_factor, y_factor, 1), order = 0)

                # cv2.imshow("", image)
                # cv2.waitKey()

                buffer[0, index] = image
                index += 1

        cap.release()
        # exit()
        label = self.one_hot(int(path[-17:-16]), 8)
        label = np.expand_dims(label, axis=0)
        return buffer, label

    def get_video_shape(self, cap):
        frame_count = 33
        # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return frame_count, frame_height, frame_width

    def get_data(self, type_of_set="train", batch_size=1):
        paths = []
        if type_of_set is "train":
            paths = self.train_set
        elif type_of_set is "val":
            paths = self.validation_set
        else:
            print('type_of_set must be "val" or "train"')

        cap = cv2.VideoCapture(paths[0])
        frame_count, frame_height, frame_width = self.get_video_shape(cap)
        cap.release()

        while True:
            batch_count = 1
            batch = np.zeros((batch_size, frame_count, self.image_y_shape, self.image_x_shape, 3), np.dtype('uint8'))
            label = np.zeros((batch_size, 8))
            for path in paths:
                batch[batch_count - 1], label[batch_count - 1] = self.video_path_to_images(path)

                if not batch_count % batch_size:
                    batch_count = 1
                    yield batch, label
                else:
                    batch_count += 1

    def get_data_length(self, type_of_set="train"):
        if type_of_set == "train":
            return len(self.train_set)
        elif type_of_set == "val":
            return len(self.validation_set)
        else:
            print("Error in class_generator.get_data_length: type_of_set must be 'val' or 'train")
