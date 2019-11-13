## make resq logo 
import numpy as np
import cv2
from scipy import ndimage
path = 'C:\\Users\\47450\\Pictures\\'
image = 'logo.png'
orig_im = cv2.imread(path+ image)
orig_im = ndimage.zoom(orig_im, (0.05,0.05,1))

shape = orig_im.shape
im = np.zeros((shape[0], shape[1]), dtype = np.str)
for i in range(shape[0]):
    for j in range(shape[1]):
        if np.sum(orig_im[i,j]) < 10:
            item = 'q'
        elif np.sum(orig_im[i,j]) < 100:
            item = 'r'
        elif np.sum(orig_im[i,j]) < 350:
            item = 'RR'
        elif np.sum(orig_im[i,j]) < 600:
            item = 'BB'
        elif np.sum(orig_im[i,j]) >= 600:
            item = '  '

        im[i][j] = item
    #im[i][j+1] = '\n'

np.savetxt(path + 'new_logo.txt', im, fmt = '%s')