import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import requests
import csv
import cv2

## yields a face from the google dataset	

#trip_type either:  'TWO_CLASS_TRIPLET', 'ONE_CLASS_TRIPLET', 'THREE_CLASS_TRIPLET', 'all'
def face_from_web_gen(path, out_shape=(224, 224), start_row=0, trip_type = 'TWO_CLASS_TRIPLET'):
    pass_ = False
    def im_reshape(orig_shape, image):
    
        factor_x = out_shape[0] / orig_shape[0]
        factor_y = out_shape[1] / orig_shape[1]

        return ndimage.zoom(image, (factor_x, factor_y, 1), order = 1)


    with open(path, "r") as f:

        csv_reader = csv.reader(f, delimiter=",")

        for row_nr, row in enumerate(csv_reader):
            # skip to start_row
            if row_nr < start_row:
                continue
            
            # triplet type 
            triplet_type = row[15]
            if trip_type == 'all':
                pass
            elif triplet_type != trip_type: # skip if not wanted
                continue


            # label - choose the label chosen by the most anotators
            anotations = [row[17], row[19], row[21], row[23], row[25], row[27]]
            label = np.argmax(np.bincount(anotations))
            ## TODO set 60% agreament

            for img_nr, row_inc in enumerate(range(3)):

                url = row[row_inc * 5]

                image = cv2.imdecode(np.frombuffer(requests.get(url).content, np.uint8), -1)

                if len(np.shape(image)) != 3:
                    pass_ = True
                    continue

                bb = []
                x1 = int(float(row[row_inc * 5 + 1]) * image.shape[1])
                x2 = int(float(row[row_inc * 5 + 2]) * image.shape[1])

                y1 = int(float(row[row_inc * 5 + 3]) * image.shape[0])
                y2 = int(float(row[row_inc * 5 + 4]) * image.shape[0])

                image = image[y1:y2, x1:x2]
                image = im_reshape(orig_shape = image.shape, image = image)

                if img_nr == 0 :
                    image1 = image.copy()
                elif img_nr == 1:
                    image2 = image.copy()
                elif img_nr == 2:
                    image3 = image.copy()

            if pass_ == False:
                yield image1, image2, image3, triplet_type, row_nr, label
            pass_ = False


if __name__ == '__main__':
    save_path = 'C:\\Users\\47450\\Documents\\ResQ Biometrics\\Data sets\\FEC_dataset\\images\\two-class_triplets\\'
    path_to_csv = 'C:\\Users\\47450\\Documents\\ResQ Biometrics\\Data sets\\FEC_dataset\\faceexp-comparison-data-train-public.csv'
    for image1, image2, image3, triplet_type, row_nr, label in face_from_web_gen(path_to_csv, start_row=1714):
        cv2.imwrite(save_path+'{}_{}_{}.jpg'.format(row_nr, '1', label), image1)
        cv2.imwrite(save_path+'{}_{}_{}.jpg'.format(row_nr, '2', label), image2)
        cv2.imwrite(save_path+'{}_{}_{}.jpg'.format(row_nr, '3', label), image3)
        print(row_nr)
        
