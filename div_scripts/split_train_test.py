import numpy as np
import os
import shutil
path_to_train = 'C:\\Users\\47450\Documents\\ResQ Biometrics\\Data sets\\ExpW\\train\\'
path_to_test = 'C:\\Users\\47450\\Documents\\ResQ Biometrics\\Data sets\\ExpW\\validation\\'
path = 'C:\\Users\\47450\\Documents\\ResQ Biometrics\\Data sets\\ExpW\\'

labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

split_ratio = 0.15

for folder in os.listdir(path):
    print('current_folder: ', folder)
    if folder not in labels:
        continue
    
    try:
        os.mkdir(path_to_train + folder)
        os.mkdir(path_to_test + folder)
    except:
        pass

    cnt = 0
    N__validation_images = int(len(os.listdir(path + folder)) * split_ratio)

    for item in os.listdir(path + folder):
        #print('moved_item: ', item, ' from folder: ', folder)

        if cnt < N__validation_images:
            shutil.move(path + folder + '\\' + item, path_to_test + folder)
        else:
            shutil.move(path + folder + '\\' + item, path_to_train + folder)
        cnt +=1


