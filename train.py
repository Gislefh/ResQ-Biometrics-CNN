import sys
sys.path.insert(0, "classes")

import numpy as np
import matplotlib.pyplot as plt
from class_model import SecModel
from class_generator import Generator
from class_utils import get_vgg16_from_keras

import keras
from keras.models import load_model
import h5py


train_path = 'C:\\Users\\47450\\Documents\\ResQ Biometrics\\Data sets\\face-expression-recognition-dataset\\images\\train'
test_path = 'C:\\Users\\47450\\Documents\\ResQ Biometrics\\Data sets\\face-expression-recognition-dataset\\images\\validation'

N_channels = 1
batch_size = 16
image_shape = (70, 70)
N_classes = 6
X_shape = (batch_size, image_shape[0], image_shape[1], N_channels)
Y_shape = (batch_size, N_classes)


gen = Generator(train_path, X_shape, Y_shape, N_classes, N_channels, batch_size)
gen.add_rotate(max_abs_angle_deg=20)
gen.add_gamma_transform(0.5,1.5)
gen.add_flip()
gen.add_shift(0.1)
batch_gen = gen.generator_from_dir(include_folder_list = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise'], N_images_per_class = 3000)

test_gen = Generator(test_path, X_shape, Y_shape, N_classes, N_channels, batch_size)
#test_gen.add_rotate(max_abs_angle_deg=20)
#test_gen.add_gamma_transform(0.5,1.5)
test_gen = test_gen.generator_from_dir(include_folder_list = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise'], N_images_per_class = 700)


### get model
#m = SecModel(N_classes)
#model = m.random_CNN(input_shape = (image_shape[0], image_shape[1], N_channels))
#model.summary()


## chaeckpoints
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
callback = []q
callback.append(early_stop)


## load old model
model = load_model("Models\\test_model.h5")


#"""
model.compile(loss='categorical_crossentropy',
          optimizer='adam',
         metrics=['acc'])

steps_per_epoch = 3000 *6/batch_size

model.fit_generator(batch_gen,
                    validation_data = test_gen,
                    steps_per_epoch = steps_per_epoch, 
                    validation_steps = 700*6/batch_size,
                    epochs = 35,
                    callbacks = callback)

model.save("Models\\test_model.h5")
exit()
#"""

### test model



#out = model.evaluate_generator(test_gen, steps = 20)
#print(model.metrics_names)
#print(out)
exit()

for x,y in batch_gen:
    for i in range(np.shape(x)[0]):
        #print('pred: ', model.predict(x[i:i+1]),'    gt: ', y[i])
        #print('Pred: ',np.argmax(model.predict(x[i:i+1])), '        GT: ', np.argmax(y[i]))
        plt.imshow(x[i, :, :, 0], cmap = 'gray')
        plt.show()
    exit()
