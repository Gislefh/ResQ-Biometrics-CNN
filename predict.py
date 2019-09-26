import sys
sys.path.insert(0, "classes")
import tensorflow as tf
import keras
import numpy as np
import cv2
from keras.models import load_model
from class_generator import Generator
from class_predict import Predict
import matplotlib.pyplot as plt

### consts
train_path = 'C:\\Users\\47450\\Documents\\ResQ Biometrics\\Data sets\\face-expression-recognition-dataset\\images\\train'
test_path = 'C:\\Users\\47450\\Documents\\ResQ Biometrics\\Data sets\\face-expression-recognition-dataset\\images\\validation'

N_channels = 1
batch_size = 16
model_shape_shape = (80, 80)
N_classes = 3
X_shape = (batch_size, model_shape_shape[0], model_shape_shape[1], N_channels)
Y_shape = (batch_size, N_classes)

## create ganerator
gen_test = Generator(test_path, X_shape, Y_shape, N_classes, N_channels, batch_size, N_images_per_class=None, class_list = ['angry', 'happy', 'neutral'])
N_data = gen_test.get_length_data()
test_gen = gen_test.flow_from_dir(set = 'test')

labels = gen_test.get_classes()

model = load_model("Models\\model_test_tensorboard_2.h5")
P = Predict(model, labels = labels)


P.pred_from_cam()
#P.conf_matrix(test_gen, N_data)
exit()




cnt_tot = 0
cnt_correct = 0
for x,y in test_gen:
    for i in range(x.shape[-1]):
        
        gt = np.argmax(y[i])
        pred = np.argmax(model.predict(x[i:i+1]))
        #print(labels[gt], labels[np.argmax(pred)])
        if pred == gt:
            cnt_correct += 1
        #plt.imshow(x[i, :, :, 0], cmap = 'gray')
        #plt.show()
        cnt_tot += 1
    print(cnt_correct/cnt_tot)
