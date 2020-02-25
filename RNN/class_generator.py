import os
from random import shuffle
import cv2
import numpy as np
from scipy.ndimage import zoom


class Generator:
    def __init__(self, path, x_shape, y_shape, n_classes, val_size):
        self.path = path
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

    def video_path_to_images(self, path):
        x_shape = 64 * 1
        y_shape = 72 * 1

        cap = cv2.VideoCapture(path)

        frame_count = 90
        # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        buffer = np.empty((1 , frame_count, y_shape, x_shape, 3), np.dtype('uint8'))

        im_shape = (frame_height, frame_width, 3)
        y_factor = y_shape / frame_height
        x_factor = x_shape / (frame_width / 2)

        index = 0
        ret = True
        while index < frame_count and ret:
            ret, image = cap.read()

            if image is not 0 and ret:
                image = image[:, int(im_shape[1] / 4):int(3 * im_shape[1] / 4)]
                image = zoom(image, (x_factor, y_factor, 1))
                buffer[0, index] = image
                index += 1
            cap.release()

        return buffer

    def get_data(self, type_of_set="train"):
        paths = []
        if type_of_set is "train":
            paths = self.train_set
        elif type_of_set is "val":
            paths = self.validation_set
        else:
            print('type_of_set must be "val" or "train"')

        while True:
            for path in paths:
                yield self.video_path_to_images(path)

    def get_data_length(self, type_of_set="train"):
        return len(self.train_set)