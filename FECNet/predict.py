import tensorflow as tf 
from generator import TripletGenerator
from model import FECNet_inceptionv3_model
from utils import distances, eval_gen
import numpy as np
import matplotlib.pyplot as plt
from custom_loss import TripletLoss
from sklearn.decomposition import PCA

class predict:

    def __init__(self, data_path, model_weight_path, N_data_samples = None, image_shape = None, model_type = 'FECNet', batch_size = 1):
        self.data_path = data_path
        self.model_weight_path = model_weight_path
        self.N_data_samples = N_data_samples
        self.image_shape = image_shape
        self.batch_size = batch_size

        if model_type == 'FECNet':
            self.embedding_size = 16
            self.model = FECNet_inceptionv3_model(input_shape = self.image_shape, embedding_size = self.embedding_size)
            self.model.load_weights(self.model_weight_path)
        
        self.__generator_FEC()
        
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
            
    def __generator_FEC(self):
        trip_gen = TripletGenerator(self.data_path, out_shape = self.image_shape, batch_size=self.batch_size, augment=False, data=self.N_data_samples, train_val_split = 0.0)
        self.generator = trip_gen.flow_from_dir()
        self.data_len =  trip_gen.get_data_len()

    #creates a list of [[pred1, im1], [pred2, im2].....]
    def __image_pred_list(self):
        pred_list = []
        im_list = np.zeros((self.data_len*3, self.image_shape[0], self.image_shape[1], self.image_shape[2]))
        for i, (x,y) in enumerate(self.generator):
            if i >= self.data_len:
                break
            predictions = np.squeeze(self.model.predict(x, batch_size = self.batch_size, steps = 1))
            for n in range(3):
                pred_list.append(predictions[n*self.embedding_size:self.embedding_size*(n+1)] )## test this
                im_list[(i*3)+n] = np.squeeze(x[n])
        return pred_list, im_list

    def __plot_2d(self, images, reduced_pred):
        size = 0.01
        fig, ax = plt.subplots(1, 1)
        ax.set_xlim(np.amin(reduced_pred[:,0])-size, np.amax(reduced_pred[:,0])+size)
        ax.set_ylim(np.amin(reduced_pred[:,1])-size, np.amax(reduced_pred[:,1])+size)        
        for i, im in enumerate(images):
            extent = [reduced_pred[i, 0]-size, reduced_pred[i, 0]+size, reduced_pred[i, 1]-size, reduced_pred[i, 1]+size]
            ax.imshow(im, extent=extent)
        plt.show()

if __name__ == "__main__":
    data_path = 'C:\\Users\\47450\\Documents\\ResQ Biometrics\\Data sets\\FEC_dataset\\images\\two-class_triplets-test'
    model_weight_path = 'Models\\FECNet_test4.h5'
    image_shape = (128, 128, 3)
    P = predict(data_path, model_weight_path, image_shape=image_shape, N_data_samples=3000)
    P.eval_gen()
    P.pca(N_comp = 2)




