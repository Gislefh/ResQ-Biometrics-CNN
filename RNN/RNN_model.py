from keras import Sequential
from keras.layers import CuDNNLSTM, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, ConvLSTM2D, LSTM, TimeDistributed, \
    GlobalAveragePooling3D, Conv3D, MaxPooling3D, AveragePooling3D


class RnnCnnModel:
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def rnnCnnModel(self, input_shape):
        model = Sequential()

        model.add(Conv3D(32, kernel_size=(4, 4, 4), input_shape=(33, input_shape[0], input_shape[1], input_shape[2])))
        model.add(Dropout(0.1))
        model.add(AveragePooling3D((2, 2, 2)))
        model.add(Conv3D(32, (4, 4, 4), activation='relu'))
        model.add(AveragePooling3D((2, 2, 2)))
        model.add(Dropout(0.1))

        model.add(ConvLSTM2D(64, (3, 3), return_sequences=True))

        model.add(Conv3D(1, (3, 3, 3), activation='sigmoid', data_format='channels_last'))
        #model.add(GlobalAveragePooling3D())
        model.add(Flatten())
        model.add(Dense(100))
        model.add(Dense(8, activation="softmax"))

        return model
