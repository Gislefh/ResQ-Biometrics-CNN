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
test_path = 'C:\\Users\\47450\\Documents\\ResQ Biometrics\\Data sets\\ExpW\\validation'

N_channels = 3
batch_size = 16
model_shape_shape = (100, 100)
N_classes = 7
X_shape = (batch_size, model_shape_shape[0], model_shape_shape[1], N_channels)
Y_shape = (batch_size, N_classes)

## create ganerator
gen_test = Generator(test_path, X_shape, Y_shape, N_classes, N_channels, batch_size, N_images_per_class=500)#, class_list = ['angry', 'happy', 'neutral'])
N_data = gen_test.get_length_data()
test_gen = gen_test.flow_from_dir(set = 'test')

labels = gen_test.get_classes()

model = load_model("Models\\model_expw_preTr_vgg16_2_cont3.h5")
P = Predict(model, labels = labels)

print(model.metrics_names)
print(model.evaluate_generator(test_gen, steps=N_data/batch_size, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1))


#P.pred_from_cam()
#P.conf_matrix(test_gen, N_data)
exit()





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
