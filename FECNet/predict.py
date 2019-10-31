import tensorflow as tf 
from generator import TripletGenerator, TripletFromOtherDataset
from model import FECNet_inceptionv3_model
from utils import distances, eval_gen
import numpy as np
import matplotlib.pyplot as plt
from custom_loss import TripletLoss
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
import cv2
from matplotlib import patches

class Predict:

    """
    Predicts on FECNet models
    """

    def __init__(self, data_path, model_weight_path, N_data_samples = None, image_shape = None, model_type = 'FECNet', batch_size = 1, generator_type = 'orig_FECNet', label_list = []):
        # Constants
        self.data_path = data_path
        self.model_weight_path = model_weight_path
        self.N_data_samples = N_data_samples
        self.image_shape = image_shape
        self.batch_size = batch_size  # should be 1 or it will crash atm
        self.label_list = label_list
        self.generator_type = generator_type
        # Load Model
        if model_type == 'FECNet':
            self.embedding_size = 16
            self.model = FECNet_inceptionv3_model(input_shape = self.image_shape, embedding_size = self.embedding_size)
            self.model.load_weights(self.model_weight_path)
        
        # Create generator
        if self.generator_type == 'orig_FECNet':
            self.__generator_FEC()
        elif self.generator_type == 'expw':
            self.__generator_expw()
        
    """
    Fits a pca model to the data and plots the result in N_comp ( now 2d, 3d not implemented)
    """
    def pca(self, N_comp = 2): 
        pred, im = self.__image_pred_list()
        pca = PCA(n_components = N_comp)
        reduced_pred = pca.fit_transform(pred)
        if N_comp == 2:
            self.__plot_2d(im, reduced_pred)
 

    """
    Prints out the accuracy of the model. 
    - Accuracy is whether the images that looks similar are colser than the other comboinations in the triplet
    """
    def eval_gen(self):
        acum = 0
        cnt = 0
        for i, (x,y) in enumerate(self.generator):
            cnt +=1
            print('predicted on ', cnt, '/ ', self.data_len, ' samples')
            pred = self.model.predict(x, batch_size = 1, steps = 1)
            d1, d2, d3 = distances(np.squeeze(pred), self.embedding_size)
            if d1 < d2 and d1 < d3:
                gt = 3
            elif d2 < d1 and d2 < d3:
                gt = 2
            elif d3 < d1 and d3 < d2:
                gt = 1
            else:
                print('what, and how!?!')
                exit()
            if gt == y:
                acum += 1
            if cnt >= self.data_len:
                break
        print('prediction accuracy on set: ', acum/cnt)


    """
    Clustering from sklearn 
    plots images with label frames 
    
    K-means:
        - args[0] = N_clusters - "numer of clusters to find"

    DBSCAN:
        -args[0] = eps -    "The maximum distance between two samples for one to be considered 
                            as in the neighborhood of the other. This is not a maximum bound 
                            on the distances of points within a cluster. (DEFAULT: 0.5)"
        -args[1] = min_samples  - "The number of samples (or total weight) in a neighborhood 
                                    for a point to be considered as a core point (DEAFULT: 5)"
    """
    def cluster(self, method = 'K-means', args = [], N_comp = 2):
        pred_list, im_list, label_list = self.__image_pred_label_list()
        if method == 'K-means':
            K = KMeans(n_clusters=args[0])
            cluster_pred = K.fit_predict(pred_list)
        elif method  == 'DBSCAN':
            C = DBSCAN(eps = args[0], min_samples = args[1])
            cluster_pred = C.fit_predict(pred_list)

        pca = PCA(n_components = N_comp)
        reduced_pred = pca.fit_transform(pred_list)

        if self.generator_type == 'expw':
            self.__plot_2d_label_frame(im_list, reduced_pred, label_list)
        
        self.__plot_2d_label_frame(im_list, reduced_pred, cluster_pred)
        plt.show()
 

      
    def __generator_FEC(self):
        trip_gen = TripletGenerator(self.data_path, out_shape = self.image_shape, batch_size=self.batch_size, augment=False, data=self.N_data_samples, train_val_split = 0.0)
        self.generator = trip_gen.flow_from_dir()
        self.data_len =  trip_gen.get_data_len()

    def __generator_expw(self):
        if self.label_list:
            data_samp = int(self.N_data_samples/ len(self.label_list))
        else:
            data_samp = int(self.N_data_samples/8)

        trip_gen = TripletFromOtherDataset(self.data_path, self.batch_size, self.image_shape, augment = False, label_list = self.label_list, N_images_per_label = data_samp)
        self.generator = trip_gen.ret_with_label()
        self.data_len = trip_gen.get_data_len()

    #creates a list of [[pred1, im1], [pred2, im2].....]
    def __image_pred_list(self):
        pred_list = []
        im_list = np.zeros((self.data_len*3, self.image_shape[0], self.image_shape[1], self.image_shape[2]))
        for i, (x,y) in enumerate(self.generator):
            if i >= self.data_len:
                break
            predictions = np.squeeze(self.model.predict(x, batch_size = self.batch_size, steps = 1))
            for n in range(3):
                pred_list.append(predictions[n*self.embedding_size:self.embedding_size*(n+1)])
                im_list[(i*3)+n] = np.squeeze(x[n])
        return pred_list, im_list

    def __image_pred_label_list(self):
        pred_list = []
        label_list = []
        im_list = np.zeros((self.data_len*3, self.image_shape[0], self.image_shape[1], self.image_shape[2]))
        for i, (x,y) in enumerate(self.generator):
            if i >= self.data_len:
                break
            predictions = np.squeeze(self.model.predict(x, batch_size = self.batch_size, steps = 1))
            for n in range(3):
                pred_list.append(predictions[n*self.embedding_size:self.embedding_size*(n+1)])
                im_list[(i*3)+n] = np.squeeze(x[n])
                if self.generator_type == 'expw':
                    label_list.append(y[n])
        return pred_list, im_list, label_list

    def __plot_2d(self, images, reduced_pred):
        size = 0.01
        fig, ax = plt.subplots(1, 1)
        ax.set_xlim(np.amin(reduced_pred[:,0])-size, np.amax(reduced_pred[:,0])+size)
        ax.set_ylim(np.amin(reduced_pred[:,1])-size, np.amax(reduced_pred[:,1])+size)        
        for i, im in enumerate(images):
            extent = [reduced_pred[i, 0]-size, reduced_pred[i, 0]+size, reduced_pred[i, 1]-size, reduced_pred[i, 1]+size]
            ax.imshow(im, extent=extent)
        plt.show()

    def __plot_2d_label_frame(self, image, reduced_pred, cluster_pred):
        size = 0.01
        
        fig, ax = plt.subplots(1, 1)
        ax.set_xlim(np.amin(reduced_pred[:,0])-size, np.amax(reduced_pred[:,0])+size)
        ax.set_ylim(np.amin(reduced_pred[:,1])-size, np.amax(reduced_pred[:,1])+size)  

        n_clusters = len(np.unique(cluster_pred))
        label_colors = []
        for i in range(len(image)):
            cmap = plt.cm.Spectral(cluster_pred[i] / n_clusters)
            label_colors.append(cmap)    

        for i, im in enumerate(image):
            extent = [reduced_pred[i, 0]-size, reduced_pred[i, 0]+size, reduced_pred[i, 1]-size, reduced_pred[i, 1]+size]
            ax.imshow(im, extent=extent)

            rect = patches.Rectangle((extent[0],extent[2]), size*2, size*2, linewidth=4, edgecolor=label_colors[i],facecolor=label_colors[i], fill=False)
            ax.add_patch(rect)

        #plt.show()

if __name__ == "__main__":
    data_path = 'C:/Users/47450/Documents/ResQ Biometrics/Data sets/FEC_dataset/images/two-class_triplets'
    model_weight_path = 'Models/FECNet_test6.h5'
    image_shape = (128, 128, 3)
    P = Predict(data_path, model_weight_path, image_shape=image_shape, N_data_samples=500, generator_type='orig_FECNet')
    #P.eval_gen()
    #P.pca(N_comp = 2)
    P.cluster(method = 'K-means', N_clusters = 10, N_comp = 2)




