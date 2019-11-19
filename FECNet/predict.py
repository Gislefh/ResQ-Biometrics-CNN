import tensorflow as tf 
from generator import TripletGenerator, TripletFromOtherDataset
from model import FECNet_inceptionv3_model, FECNet_inceptionv3_dense_model
from utils import Distances, eval_gen
import numpy as np
import matplotlib.pyplot as plt
from custom_loss import TripletLoss
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
import cv2
from matplotlib import patches
from scipy.spatial import distance
class Predict:

    """
    Predicts on FECNet models

    generator_type: 'orig_FECNet' or 'expw'. orig_FECNet asumes that all the images are stored in one folder, while expw assumes that the images are stroed in there repectve labeled folder 
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
        elif model_type == 'FECNet_dense':
            self.embedding_size = 16
            self.model = FECNet_inceptionv3_dense_model(input_shape = self.image_shape, embedding_size = self.embedding_size)
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
            d1, d2, d3 = Distances(np.squeeze(pred), self.embedding_size)
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
                            on the distances of points within a cluster. (DEFAULT: 0.5 but should probably be similar to delta_trip_loss)"
        -args[1] = min_samples  - "The number of samples (or total weight) in a neighborhood 
                                    for a point to be considered as a core point (DEAFULT: 5)"
    """
    def cluster(self, method = 'K-means', args = []):
        # Clustering
        pred_list, im_list, label_list = self.__image_pred_label_list()
        if method == 'K-means':
            K = KMeans(n_clusters=args[0])
            cluster_pred = K.fit_predict(pred_list)
        elif method  == 'DBSCAN':
            C = DBSCAN(eps = args[0], min_samples = args[1])
            cluster_pred = C.fit_predict(pred_list)
        else:
            raise Exception('-- FROM SELF -- Choose from list of clustering algorithms')

        ## 2D
        pca = PCA(n_components = 2)
        reduced_pred = pca.fit_transform(pred_list)

        if self.generator_type == 'expw':
            self.__plot_2d_label_frame(im_list, reduced_pred, label_list)
        
        self.__plot_2d_label_frame(im_list, reduced_pred, cluster_pred)

        ## 3D 
        pca = PCA(n_components = 3)
        reduced_pred = pca.fit_transform(pred_list)
        self.__plot_3d(reduced_pred, label_list)
        self.__plot_3d(reduced_pred, cluster_pred)   
             
        # Show all
        plt.show()
    
    #guess it will be slow, at least for seach_space_len with large values
    def query_for_sim_expressions(self, image, seach_space_len = 3000, N_images_to_show = 10):

        #cnt = 0
        for cnt, (x, y) in enumerate(self.generator):
            '''
            plt.figure(1)
            plt.imshow(np.squeeze(x[0]))
            plt.figure(2)
            plt.imshow(np.squeeze(x[1]))
            plt.figure(3)
            plt.imshow(np.squeeze(x[2]))
            plt.show()
            '''
            # Init
            if (cnt == 0) and (image == None):
                image_w_extra_dim = x[0].copy()
                im = image_w_extra_dim[0]

                # Find predicted value of image
                image_plus_zeros = [image_w_extra_dim, np.zeros(image_w_extra_dim.shape), np.zeros(image_w_extra_dim.shape)]
                im_pred = self.model.predict(image_plus_zeros, batch_size = 1, steps = 1)[0, 0:self.embedding_size]
                
                best_im = np.zeros((N_images_to_show, im.shape[0], im.shape[1], im.shape[2]))
                best_dist = [np.inf] * N_images_to_show
                continue

            print(cnt)

            for i in range(3):
                if np.array_equal(x[i], image_w_extra_dim):
                    x[i] = np.zeros(image_w_extra_dim.shape)
                  
            for j in range(N_images_to_show): #slow?
                for k in range(3):
                    if np.array_equal(best_im[j], np.squeeze(x[k])):
                        x[k] = np.zeros(image_w_extra_dim.shape)


            pred = np.squeeze(self.model.predict(x, batch_size = 1, steps = 1))

            for i in range(3):
                
                d = pred[i*16: (i*16)+16]
                dist = distance.euclidean(im_pred, d)

                for j in range(N_images_to_show):
                    if dist < best_dist[j]:
                        best_dist[j] = dist
                        best_im[j] = np.squeeze(x[i].copy())
                        break
            
            if cnt >= seach_space_len:
                break

        # Plots
        result = np.append(best_im[0], best_im[1], axis = 1)  
        result_dist = [1- best_dist[0], 1- best_dist[1]]
        for i in range(2, N_images_to_show):
            result = np.append(result, best_im[i], axis = 1)
            result_dist.append(1-best_dist[i])
        print(result_dist)
        plt.figure(1)
        plt.subplot(2,2,1)
        plt.imshow(im)
        plt.subplot(2,2,2)
        plt.imshow(result)
        plt.subplot(2,2,4)
        plt.plot(list(range(len(result_dist))), result_dist, 'b|-')
        plt.ylabel('Similarity')
        plt.grid()
        plt.show()
            
        



      
    def __generator_FEC(self):
        trip_gen = TripletGenerator(self.data_path, out_shape = self.image_shape, batch_size=self.batch_size, augment=False, data=self.N_data_samples, train_val_split = 0.05)
        self.generator = trip_gen.flow_from_dir(set = 'train')
        self.data_len =  trip_gen.get_data_len(set = 'train')


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
            if cluster_pred[i] == -1:
                label_colors.append([0,0,0,1])
            cmap = plt.cm.Spectral(cluster_pred[i] / n_clusters)
            label_colors.append(cmap)    

        for i, im in enumerate(image):
            extent = [reduced_pred[i, 0]-size, reduced_pred[i, 0]+size, reduced_pred[i, 1]-size, reduced_pred[i, 1]+size]
            ax.imshow(im, extent=extent)

            rect = patches.Rectangle((extent[0],extent[2]), size*2, size*2, linewidth=4, edgecolor=label_colors[i],facecolor=label_colors[i], fill=False)
            ax.add_patch(rect)

    def __plot_3d(self, input_vec, cluster_pred):
        n_clusters = len(np.unique(cluster_pred))
        label_colors = []
        for i in range(len(input_vec)):
            if cluster_pred[i] == -1:
                label_colors.append([0,0,0,1])
            cmap = plt.cm.Spectral(cluster_pred[i] / n_clusters)
            label_colors.append(cmap) 

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xs =input_vec[:, 0], ys = input_vec[:, 1], zs = input_vec[:, 2], zdir='z', s=20, c=label_colors)


if __name__ == "__main__":

    # Make GPU invis to tf
    #import os 
    #os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
    #import tensorflow as tf 

    # Data
    #data_path = 'C:/Users/47450/Documents/ResQ Biometrics/Data sets/ExpW/train' # ExpW
    #data_path = 'C:/Users/47450/Documents/ResQ Biometrics/Data sets/ExpW/test' # fecnet data
    data_path = 'C:/Users/47450/Documents/ResQ Biometrics/Data sets/FEC_dataset/images/two-class_triplets'
    # Model
    model_weight_path = 'Models/FECNet_dense_1.h5'
    image_shape = (224, 224, 3)
    P = Predict(data_path, model_weight_path, image_shape=image_shape, N_data_samples=30000, generator_type='orig_FECNet', model_type='FECNet_dense')
    #P.eval_gen()
    #P.pca(N_comp = 2)
    #P.cluster(method = 'K-means', args = [12])
    P.query_for_sim_expressions(None)




