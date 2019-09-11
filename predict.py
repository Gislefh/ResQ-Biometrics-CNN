import sys
sys.path.insert(0, "classes")

import keras
import numpy as np
import cv2
from keras.models import load_model
from class_generator import Generator
from class_predict import Predict
import matplotlib.pyplot as plt

### consts
test_path = 'C:\\Users\\47450\\Documents\\ResQ Biometrics\\Data sets\\face-expression-recognition-dataset\\images\\validation'
from_web_path = 'C:\\Users\\47450\\Documents\\ResQ Biometrics\\Data sets\\Data_set_from_web'
path_to_google_data_cvs =  'C:\\Users\\47450\\Documents\\ResQ Biometrics\\Data sets\\FEC_dataset\\faceexp-comparison-data-train-public.csv'
N_channels = 1
batch_size = 16
model_shape_shape = (70, 70)
N_classes = 7
X_shape = (batch_size, model_shape_shape[0], model_shape_shape[1], N_channels)
Y_shape = (batch_size, N_classes)

## create ganerator
#gen = Generator(path_to_google_data_cvs, X_shape, Y_shape, N_classes, N_channels, batch_size)
#gen = gen.generator_from_dir(include_folder_list = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise'], N_images_per_class = 700)
#W_gen = gen.face_from_web_gen()

gen_test = Generator(test_path, X_shape, Y_shape, N_classes, N_channels, batch_size)
N_data = gen_test.get_length_data()
test_gen = gen_test.flow_from_dir(set = 'test')




model = load_model("Models\\model3.h5")
P = Predict(model, labels = ['angry','disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'])

#P.pred_from_cam()
P.conf_matrix(test_gen, N_data)
exit()
label_list = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']



"""  Display images from google dataset with predictions 
cnt = 0
for image in W_gen:
    
    if cnt < 300:
        print(cnt)
        cnt += 1
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orig_im = gray.copy()
    gray = np.clip(gray / 255, 0, 1)

    gray = np.expand_dims(gray, 0)
    gray = np.expand_dims(gray, -1)

    pred = model.predict(gray)
    pred_label = label_list[np.argmax(pred)]
    conf = pred[0, np.argmax(pred)]
    if conf < 0.3:
        print(conf)
        continue

    plt.figure('label: '+pred_label +'    confidence: '+ str(conf))
    plt.imshow(orig_im, cmap = 'gray')
    plt.show()

"""

