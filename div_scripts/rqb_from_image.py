## make resq logo 
import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot  as plt
path = 'C:\\Users\\47450\\Pictures\\'
image = 'rek_logo.jpg'
orig_im = cv2.imread(path+ image)
orig_im = ndimage.zoom(orig_im, (3,3,1), order = 5)
#orig_im = cv2.cvtColor(orig_im, cv2.COLOR_BGR2RGB)

plt.figure(1)
plt.imshow(orig_im)
orig_im[orig_im>240] = 255
plt.figure(2)
plt.imshow(orig_im)
plt.show()

cv2.imwrite(path + '/'+ 'new_rek_logo.jpg', orig_im)