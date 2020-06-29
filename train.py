import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from datetime import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sb
from scipy.ndimage import zoom
import cv2
import dlib
from imutils.face_utils import rect_to_bb

from classes.class_generator import Generator
from classes.class_utils import get_inception_w_imnet
from classes.class_model import SecModel


train_path = '../Data sets/FEC_dataset/Predictions/Fecnet_class23_for_align/cluster_from_frePlus_removed_wrong'
new_model_name = 'from_ferplus_v2.h5'
save_model_path = 'models\\'

if new_model_name in os.listdir(save_model_path):
    print('Model name exists. Change the model name')
    exit()
# consts
N_channels = 3
N_images_per_class = 2000
batch_size = 128
image_shape = (224, 224)
val_size = 0.2
labels_to_use = ['anger', 'happiness', 'neutral', 'sadness', 'surprise']
#labels_to_use = list([str(x) for x in range(35)])#n first clusters



N_classes = len(labels_to_use)
X_shape = (batch_size, image_shape[0], image_shape[1], N_channels)
Y_shape = (batch_size, N_classes)


gen_train = Generator(train_path,
                      X_shape,
                      Y_shape,
                      N_classes,
                      N_channels,
                      batch_size,
                      train_val_split=val_size,
                      N_images_per_class=N_images_per_class,
                      class_list=labels_to_use)

gen_train.add_noise(0.22)
gen_train.add_rotate(max_abs_angle_deg=30)
gen_train.add_gamma_transform(0.5, 1.5)
gen_train.add_flip()

train_gen = gen_train.flow_from_dir(set='train', crop=False)
val_gen = gen_train.flow_from_dir(set='val', augment_validation=False, crop=False)


model = get_inception_w_imnet(input_shape=(image_shape[0], image_shape[1], N_channels), N_classes=N_classes,
                              freeze_layers=True)
#M = SecModel(N_classes=len(labels_to_use))
#model = M.random_CNN(input_shape=(image_shape[0], image_shape[1], N_channels))


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

steps_per_epoch = int(np.floor(gen_train.get_length_data() * (1 - val_size)) / batch_size)
val_steps_per_epoch = int(np.floor(gen_train.get_length_data() * val_size) / batch_size)




def show_confusion_matrix(epoch, show = True):
    global cnt, x, y
    acc_count = []
    conf_matrix = np.zeros((N_classes, N_classes))
    cnt = 0
    for x, y in val_gen:
        if cnt >= val_steps_per_epoch * batch_size:
            break
        preds = np.argmax(loaded_model.predict(x), axis=1)
        gt = np.argmax(y, axis=1)
        for i in range(x.shape[0]):
            if preds[i] == gt[i]:
                acc_count.append(1)
            else:
                acc_count.append(0)
            conf_matrix[preds[i], gt[i]] = conf_matrix[preds[i], gt[i]] + 1
            cnt += 1
        
        print(cnt, '/', val_steps_per_epoch * batch_size, end='\r')
    print(sum(acc_count)/len(acc_count))
    if labels_to_use:
        df = pd.DataFrame(conf_matrix, index=labels_to_use, columns=labels_to_use)
    else:
        df = pd.DataFrame(conf_matrix, index=os.listdir(train_path), columns=os.listdir(train_path))
    
    plt.figure(epoch)
    sb.heatmap(df, annot=True)
    if show:
        plt.show()

def train_model():
    best_acc = 0
    early_stop = 0
    global cnt, metrics_names, best_loss
    for epoch in range(300):
        if early_stop > 10:
            break
        model.reset_metrics()
        for cnt, (image_batch, label_batch) in enumerate(train_gen):
            result = model.train_on_batch(image_batch, label_batch, reset_metrics=False)
            metrics_names = model.metrics_names
            print("batch: ",
                  "{}/{}  ".format(cnt, steps_per_epoch),
                  "train: ",
                  "{}: {:.3f}".format(metrics_names[0], result[0]),
                  "{}: {:.3f}".format(metrics_names[1], result[1]), end = '\r')
            
            if cnt >= steps_per_epoch:
                break

        for cnt, (image_batch, label_batch) in enumerate(val_gen):
            
            result = model.test_on_batch(image_batch, label_batch,
                                         reset_metrics=False)
            if cnt >= val_steps_per_epoch:
                break

        metrics_names = model.metrics_names
        print("\neval: ",
            "{}: {:.3f}".format(metrics_names[0], result[0]),
            "{}: {:.3f}".format(metrics_names[1], result[1]))
            

        if best_acc < result[1]:
            best_acc = result[1]
            model.save(save_model_path + '/' + str(epoch)+ '_' + new_model_name , overwrite=True, save_format='h5')
            #show_confusion_matrix(epoch)
            early_stop = 0
        else:
            early_stop +=1
        
def crop_pred_save(model, labels_in_order, data_path, save_image_shape, save_path):  
    detector = dlib.get_frontal_face_detector()
    input_shape = model.input_shape
    cnt = 0
    for folder in os.listdir(data_path):
        print('working on folder: ', folder)
        for image_path in os.listdir(data_path +'/'+ folder):
            # to gray
            image = cv2.imread(data_path +'/'+ folder +'/'+ image_path) 
            if not np.shape(image):
                continue
            im_to_model = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # crop
            rects = detector(im_to_model, 1)
            if not rects:
                continue
            (x, y, w, h) = rect_to_bb(rects[0])
            
            if w < 10 or h < 10: # either rotated face or too small
                continue
            if x < 0 or y < 0: #bb out of bounds
                continue 
            
            im_to_model = im_to_model[y:h+y, x:w+x]

            # zoom to fit model
            factor_x = input_shape[1] / im_to_model.shape[0]
            factor_y = input_shape[2] / im_to_model.shape[1]
            im_to_model = zoom(im_to_model, (factor_x, factor_y), order=1)

            # pred
            im_to_model = np.clip(im_to_model / 255, 0, 1)
            im_to_model = np.expand_dims(im_to_model, axis =0)
            im_to_model = np.expand_dims(im_to_model, axis =-1)
            prediction = model.predict(im_to_model)
            
            pred_max = np.squeeze(np.argmax(prediction))
            if prediction[0, pred_max] < 0.95:
                continue
            pred_label = labels_in_order[pred_max]

            #crop orig
            #(x, y, w, h) = rect_to_bb(rects[0])
            image = image[y:h+y, x:w+x]

            # zoom original
            factor_x = save_image_shape[0] / image.shape[0]
            factor_y = save_image_shape[1] / image.shape[1]
            image = zoom(image, (factor_x, factor_y, 1), order=1)
            
            # save 
            if pred_label not in os.listdir(save_path):
                os.mkdir(save_path +'/'+ pred_label)
            cv2.imwrite(save_path +'/'+ pred_label +'/'+ str(cnt)+'.jpg', image)
            cnt +=1

def pred_resave(model, labels_in_order, data_path, save_path):
    input_shape = model.input_shape
    cnt = 0
    for folder in os.listdir(data_path):
        for image_name in os.listdir(data_path +'/'+ folder):

            orig_image = cv2.imread(data_path +'/'+ folder +'/'+ image_name) 
            if not np.shape(orig_image):
                continue
            image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

            # zoom to fit model
            if np.shape(image) != (input_shape[1], input_shape[2], input_shape[3]):
                factor_x = input_shape[1] / image.shape[0]
                factor_y = input_shape[2] / image.shape[1]
                image = zoom(image, (factor_x, factor_y, 1), order=1)

            # pred
            image = np.clip(image / 255, 0, 1)
            image = np.expand_dims(image, axis =0)
            prediction = model.predict(image)
            pred_max = np.squeeze(np.argmax(prediction))
            pred_label = labels_in_order[pred_max]

            # save 
            if pred_label not in os.listdir(save_path):
                os.mkdir(save_path +'/'+ pred_label)
            success = cv2.imwrite(save_path +'/'+ pred_label +'/'+ str(cnt)+'.jpg', orig_image)
            if not success:
                print('not able to save image')
            cnt +=1

def remove_wrong_labels(model, labels_in_order, data_path, save_path):
    input_shape = model.input_shape
    cnt = 0
    for folder in os.listdir(data_path):
        
        for im_seen_cnt, image_name in enumerate(os.listdir(data_path +'/'+ folder)):
            if im_seen_cnt > 10000:
                break
            orig_image = cv2.imread(data_path +'/'+ folder +'/'+ image_name) 
            if not np.shape(orig_image):
                continue
            image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

            # zoom to fit model
            if np.shape(image) != (input_shape[1], input_shape[2], input_shape[3]):
                factor_x = input_shape[1] / image.shape[0]
                factor_y = input_shape[2] / image.shape[1]
                image = zoom(image, (factor_x, factor_y, 1), order=1)

            # pred
            image = np.clip(image / 255, 0, 1)
            image = np.expand_dims(image, axis =0)
            prediction = model.predict(image)
            pred_max = np.squeeze(np.argmax(prediction))
            pred_label = labels_in_order[pred_max]
            if pred_label == folder:
                if pred_label not in os.listdir(save_path):
                    os.mkdir(save_path +'/'+ pred_label)
                success = cv2.imwrite(save_path +'/'+ pred_label +'/'+ str(cnt)+'.jpg', orig_image)
                if not success:
                    print('not able to save image')
                print(cnt, end = '\r')
                cnt +=1


loaded_model = tf.keras.models.load_model('models/38_from_ferplus_v2.h5')
show_confusion_matrix(7)

#data_path = '../Data sets/FEC_dataset/Predictions/Fecnet_class23_for_align/cluster_from_ferPlus_cnn'
#save_path = '../Data sets/FEC_dataset/Predictions/Fecnet_class23_for_align/cluster_from_frePlus_removed_wrong'

#pred_resave(loaded_model, labels_to_use, data_path, save_path)
#remove_wrong_labels(loaded_model, labels_to_use, data_path, save_path)
#train_model()


