from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

import matplotlib.pyplot as plt
import numpy  as np

import sys
sys.path.insert(0, "classes")
from class_utils import one_hot



(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = np.expand_dims(x_train, axis = 3)
x_test = np.expand_dims(x_test, axis = 3)



x_shape = x_train.shape
Y_train = np.zeros((len(y_train), np.amax(y_train)))

for i in range(len(y_train)):
    Y_train[i, :] = one_hot(y_train[i]-1, np.amax(y_train))


Y_test = np.zeros((len(y_test), np.amax(y_test)))
for i in range(len(y_test)):
    Y_test[i, :] = one_hot(y_test[i]-1, np.amax(y_test))

    
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape = (28,28, 1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.1))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.1))

#M.add(Conv2D(128, (3, 3), activation='relu'))
#M.add(Conv2D(128, (3, 3), activation='relu'))
#M.add(MaxPooling2D(pool_size=(2, 2)))
#M.add(Dropout(0.1))

#M.add(Conv2D(255, (3, 3), activation='relu'))
#M.add(Conv2D(255, (3, 3), activation='relu'))
#M.add(MaxPooling2D(pool_size=(2, 2)))
#M.add(Dropout(0.1))

#M.add(Conv2D(512, (3, 3), activation='relu'))
#M.add(Conv2D(512, (3, 3), activation='relu'))
#M.add(MaxPooling2D(pool_size=(2, 2)))
#M.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(np.amax(y_train), activation='softmax'))



model.summary()


model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['acc'])



model.fit(x=x_train, y=Y_train, batch_size=32, epochs=5, verbose=1, callbacks=None)


for i in range(20):
    pred = model.predict(x_test[i:i+1])
    plt.imshow(x_test[i, :, :, 0], cmap = 'gray')
    print('gjett:  ',y_test[i], '     fasit:  ',np.argmax(pred)+1)
    plt.show()