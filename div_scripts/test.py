'''
Testing what takes time in the generator
'''


import sys
sys.path.insert(0, "C:\\Users\\47450\\Documents\\ResQ Biometrics\\ResQ-Biometrics-CNN")
import numpy as np
from class_generator import Generator

##conts 
train_path = 'C:\\Users\\47450\Documents\\ResQ Biometrics\\Data sets\\Expw_and_FRE_ch\\train'
N_channels = 1
N_images_per_class = 30000
batch_size = 16
image_shape = (48, 48)
N_classes = 2
X_shape = (batch_size, image_shape[0], image_shape[1], N_channels)
Y_shape = (batch_size, N_classes)
val_size = 0.3


### generator
gen_train = Generator(train_path, X_shape, Y_shape, N_classes, N_channels, batch_size, class_list=['happy', 'neutral'], train_val_split=val_size, N_images_per_class=N_images_per_class)
gen_train.add_rotate(max_abs_angle_deg=20)
gen_train.add_gamma_transform(0.5,1.5)
gen_train.add_flip()
gen_train.add_shift(0.1)
#gen_train.add_zoom(zoom_range= [0.2,2])

train_gen = gen_train.flow_from_dir(set = 'train')

for x,y in train_gen:
    exit()