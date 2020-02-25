from keras import Sequential
from keras.layers import CuDNNLSTM, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, ConvLSTM2D, LSTM, TimeDistributed, GlobalAveragePooling3D ,Conv3D


class RnnCnnModel:
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def rnnCnnModel(self, input_shape):
        model = Sequential()
        model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                             input_shape=(100, input_shape[0], input_shape[1], input_shape[2]),
                             padding='same', return_sequences=True))

        model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                             padding='same', return_sequences=True))

        model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                             padding='same', return_sequences=True))

        model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                             padding='same', return_sequences=True))

        model.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
                         activation='sigmoid',
                         padding='same', data_format='channels_last'))
        model.add(GlobalAveragePooling3D())
        model.add(Dense(100))
        model.add(Dense(8, activation="softmax"))
        model.compile(loss='binary_crossentropy', optimizer='adam')

        return model
