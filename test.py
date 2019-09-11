import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
save_path = 'C:\\Users\\47450\\Documents\\ResQ Biometrics\\Data sets\\ExpW\\'
data_path = 'C:\\Users\\47450\\Downloads\\ExpW\\origin\\'
label_path = 'C:\\Users\\47450\\Downloads\\'

label_list = list(open(label_path + 'label.lst'))

""" Label_list is saved as:
image_name   face_id_in_image    face_box_top   face_box_left    face_box_right    face_box_bottom     face_box_cofidence     expression_label
"""

labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
#files = os.listdir(path):

for im_number, item in enumerate(label_list):
    impath = data_path + item.split(' ')[0]
    label = item.split(' ')[-1]
    bb = item.split(' ')[2:6]
    label_name = labels[int(label)]
    im = cv2.imread(impath)
 
    im = im[int(bb[0]):int(bb[3]), int(bb[1]):int(bb[2])]
    
    if label_name not in os.listdir(save_path):
        os.mkdir(save_path + label_name)
    
    cv2.imwrite(save_path + label_name + '\\' +str(im_number)+'.jpg', im)

    print('image ',im_number, 'of ', len(label_list), ' saved')


    



