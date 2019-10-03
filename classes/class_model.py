import keras
import tensorflow
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, GlobalMaxPooling2D, Conv2D, MaxPooling2D, Input


class SecModel:

    def __init__(self, N_classes):
        self.N_classes = N_classes
        self.model = Sequential()
        self.set_filter_list = []

    # def set_filters(self, ):

    """
    returns a squential model

    IN:
        input_shape: shape of the data coming in
        kernel_shape: int or tuple
        dropout: depth of network, int
        depth: 

    """

    def dynamic_model(self, input_shape, kernel_sizes, depth, dropout=0.1, activation_conv='relu',
                      activation_out='softmax', max_pool_shape=(2, 2)):

        self.model.add(Conv2D(2 ** 5, (kernel_sizes, kernel_sizes), activation='relu', input_shape=input_shape))
        self.model.add(Conv2D(2 ** 5, (kernel_sizes, kernel_sizes)))
        self.model.add(MaxPooling2D(pool_size=max_pool_shape))
        # Convolution
        for layer in range(len(depth - 1)):
            filters = 2 ** (layer + 6)
            self.model.add(Conv2D(filters, (kernel_sizes, kernel_sizes)))
            self.model.add(Conv2D(filters, (kernel_sizes, kernel_sizes)))
            self.model.add(MaxPooling2D(pool_size=max_pool_shape))
            if dropout:
                self.model.add(Dropout(dropout))

        # Dense

        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(self.N_classes, activation='softmax'))

        return self.model

    def compile_model(self, loss='categorical_crossentropy', optimizer='adam', metrics=['acc']):

        self.model.compile(loss=loss,
                           optimizer=optimizer,
                           metrics=metrics)

        return self.model

    def generate_CNN(self,
                     input_shape,
                     layers,
                     filter_size,
                     filter_size_fl,
                     filters_fl,
                     filters,
                     dropout_rate,
                     ):

        M = Sequential()
        M.add(Conv2D(np.ceil(filters_fl), (np.ceil(filter_size_fl), np.ceil(filter_size_fl)), activation='relu',
                     input_shape=input_shape))
        for i in range(layers):
            M.add(Conv2D(np.ceil(filters * i), (np.ceil(1 * filter_size), np.ceil(1 * filter_size)), activation='relu'))
			# M.add(Conv2D(np.ceil(filters*2*i), (np.ceil(1 * filter_size), np.ceil(1 * filter_size)), activation='relu'))
            M.add(MaxPooling2D(pool_size=(2, 2)))
            M.add(Dropout(dropout_rate))

     #   while (input > filter_size):
     #       input = (input - (filter_size - 1)) / 2

        return M

    def random_CNN(self, input_shape):
        M = Sequential()
        M.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        M.add(MaxPooling2D(pool_size=(2, 2)))

        M.add(Conv2D(64, (3, 3), activation='relu'))
        M.add(MaxPooling2D(pool_size=(2, 2)))

        M.add(Dropout(0.1))

        M.add(Conv2D(128, (3, 3), activation='relu'))
        M.add(MaxPooling2D(pool_size=(2, 2)))

        M.add(Conv2D(255, (3, 3), activation='relu'))
        M.add(Dropout(0.1))
        # M.add(MaxPooling2D(pool_size=(2, 2)))

        # M.add(Conv2D(512, (3, 3), activation='relu'))
        # M.add(GlobalMaxPooling2D())

        # M.add(Conv2D(128, (3, 3), activation='relu'))
        # M.add(Dropout(0.05))

        # M.add(Conv2D(128, (3, 3), activation='relu'))
        # M.add(Conv2D(128, (3, 3), activation='relu'))
        # M.add(MaxPooling2D(pool_size=(2, 2)))
        # M.add(Dropout(0.05))

        # M.add(Conv2D(255, (3, 3), activation='relu'))
        # M.add(Conv2D(255, (3, 3), activation='relu'))
        # M.add(MaxPooling2D(pool_size=(2, 2)))
        # M.add(Dropout(0.1))

        # M.add(Conv2D(512, (3, 3), activation='relu'))
        # M.add(Conv2D(512, (3, 3), activation='relu'))
        # M.add(MaxPooling2D(pool_size=(2, 2)))
        # M.add(Dropout(0.1))

        M.add(Flatten())
        # M.add(Dense(255, activation='relu'))
        # M.add(Dropout(0.1))
        M.add(Dense(512, activation='relu'))
        M.add(Dropout(0.1))
        M.add(Dense(self.N_classes, activation='softmax'))

        return M


class NonSecModel:

    def __init__(self, input_shape, N_classes):
        self.input_shape = input_shape
        self.N_classes = N_classes

    def test_model(self):
        input_ = Input(shape=self.input_shape)

        for i in range(layers):
            x = Conv2D(32 * filters, (1 * filter_size, 1 * filter_size), activation='relu',
                       input_shape=self.input_shape)(input_)

            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Dropout(dropout_rate)(x)

        x = Conv2D(64, (4, 4), activation='relu')(x)
        # x = MaxPooling2D(pool_size=(2, 2))(x)
        # x = Dropout(0.1)(x)

        # x = Conv2D(128, (4, 4), activation='relu')(x)
        # x = MaxPooling2D(pool_size=(2, 2))(x)
        # x = Dropout(0.1)(x)

        # x = Conv2D(255, (4, 4), activation='relu')(x)
        # x = Dropout(0.1)(x)
        # x = GlobalMaxPooling2D()(x)

        # x = Dense(512, activation='relu')(x)
        # x = Dropout(0.1)(x)
        # x = Dense(self.N_classes, activation='softmax')(x)

        self.model = Model(inputs=input_, outputs=x)
        return self.model
