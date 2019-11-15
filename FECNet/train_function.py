from generator import TripletGenerator
from model import faceNet_inceptionv3_model, test_siam_model, FECNet_inceptionv3_model, FECNet_inceptionv3_dense_model
from custom_loss import TripletLoss
from custom_metrics import CustomMetrics
import numpy as np
#import keras
from tensorflow import keras
import os


class Train:

    def __init__(self, data_path, save_model_path, new_model_name, image_shape, batch_size, delta_trip_loss,
                    embedding_size=16, N_data_samples=None, model_type='FECNet',
                    callback_list=[], augment_data = True, optimizer = 'adam', train_val_spilt = 0.3,
                    load_weights = None, import_model = None, dir_or_mem = 'dir'):
        # Constants
        self.data_path = data_path              
        self.save_model_path = save_model_path
        self.new_model_name = new_model_name
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.N_data_samples = N_data_samples
        self.model_type = model_type
        self.augment_data = augment_data
        self.optimizer = optimizer
        self.callback_list = callback_list
        self.callbacks = []
        self.delta_trip_loss = delta_trip_loss
        self.train_val_spilt = train_val_spilt
        self.load_weights = load_weights  #None or path to weights
        self.dir_or_mem = dir_or_mem # flow from dir ( 'dir' ) or from memory ('mem') 
        
        # Tests
        if new_model_name in os.listdir(save_model_path):
            raise Exception('--FROM SELF--: Model name exists. Change the model name')
        if (self.image_shape[0] < 100 or self.image_shape[1] < 100) and model_type != 'siamTest':
            raise Exception('--FROM SELF--: Input images are too small')
        if self.image_shape[2] != 3:
            raise Exception('--FROM SELF--: Input images should have 3 channels')
        
        # Functions
        if import_model:
            self.model = import_model
        else:
            self.__load_model()
        self.__create_generator()
        self.__custom_loss_and_metrics()
        self.__callbacks()

            

    def __load_model(self):
        if self.model_type == 'FECNet':
            self.model = FECNet_inceptionv3_model(input_shape=self.image_shape, embedding_size=self.embedding_size) 
        elif self.model_type == 'FaceNet':
            self.model = faceNet_inceptionv3_model(input_shape=self.image_shape, embedding_size=self.embedding_size)
        elif self.model_type == 'siamTest':
            self.model = test_siam_model(input_shape=self.image_shape, embedding_size=self.embedding_size)
        elif self.model_type == 'FECNet_dense':
            self.model = FECNet_inceptionv3_dense_model(input_shape=self.image_shape, embedding_size=self.embedding_size)

        else:
            raise Exception('--FROM SELF--: No model type selected')
        
        # Load Weights
        if self.load_weights:
            self.model.load_weights(self.load_weights)
            
        self.model.summary()
        
    
    def __create_generator(self):
        
        trip_gen = TripletGenerator(self.data_path, out_shape = self.image_shape, batch_size=self.batch_size, augment=True, data = self.N_data_samples, train_val_split=self.train_val_spilt)
        
        if self.dir_or_mem == 'dir':
            # Training 
            self.train_generator = trip_gen.flow_from_dir(set = 'train')
            self.data_len_train = trip_gen.get_data_len(set = 'train')
            # Validation
            self.val_generator = trip_gen.flow_from_dir(set = 'val')
            self.data_len_val = trip_gen.get_data_len(set = 'val')

        elif self.dir_or_mem == 'mem':
            # Training 
            self.train_generator = trip_gen.flow_from_mem(set = 'train')
            self.data_len_train = trip_gen.get_data_len(set = 'train')
            # Validation
            self.val_generator = trip_gen.flow_from_mem(set = 'val')
            self.data_len_val = trip_gen.get_data_len(set = 'val')
            

    def __custom_loss_and_metrics(self):
        # Custom loss
        L = TripletLoss(delta=self.delta_trip_loss, embedding_size=self.embedding_size)
        loss_fun = L.trip_loss

        # Custom Metrics - TODO make it work
        #M = CustomMetrics(embedding_size=embedding_size)
        #acc_fun = M.triplet_accuracy

        self.model.compile(loss=loss_fun,
                    optimizer=self.optimizer)

    def __callbacks(self):
        save_best = keras.callbacks.ModelCheckpoint(self.save_model_path + self.new_model_name,
                                            monitor='val_loss',
                                            verbose=1,
                                            save_best_only=True,
                                            save_weights_only=True,
                                            mode='min',
                                            period=1)

        tensorboard_name = 'tensorboard_' + self.new_model_name[:-3]
        tensorboard = keras.callbacks.TensorBoard(log_dir=self.save_model_path + tensorboard_name, 
                                                histogram_freq=0, 
                                                batch_size=self.batch_size, 
                                                write_graph=True, 
                                                write_grads=True, 
                                                write_images=False, 
                                                embeddings_freq=0, 
                                                embeddings_layer_names=None, 
                                                embeddings_metadata=None, 
                                                embeddings_data=None, 
                                                update_freq='epoch')                                    

        if 'save_best' in self.callback_list:
            self.callbacks.append(save_best)
        if 'tensorboard' in self.callback_list:
            self.callbacks.append(tensorboard)


    def fit(self):
        steps_per_epoch = self.data_len_train / self.batch_size
        validation_steps = self.data_len_val / self.batch_size
        
        self.model.fit_generator(self.train_generator, steps_per_epoch=steps_per_epoch, 
                                    validation_data=self.val_generator, validation_steps=validation_steps,
                                    epochs=200, shuffle=False, callbacks=self.callbacks)


'''
if __name__ == '__main__':
    data_path = 'C:\\Users\\47450\\Documents\\ResQ Biometrics\\Data sets\\FEC_dataset\\images\\two-class_triplets'
    image_shape = (128, 128, 3)
    delta_trip_loss = 0.1
    embedding_size = 16 # faceNet uses 128, FECNet uses 16.
    batch_size = 4
    save_model_path = 'C:\\Users\\47450\\Documents\\ResQ Biometrics\\ResQ-Biometrics-CNN\\FECNet\\Models'
    new_model_name = 'FECNet_test5.h5'

    T = Train(data_path, save_model_path, new_model_name, image_shape, batch_size, delta_trip_loss,
                    embedding_size=16, N_data_samples=30000, model_type='FECNet',
                    callback_list=['tensorboard', 'save_best'], augment_data = True, optimizer = 'adam')

    T.fit()

'''


if __name__ == '__main__':

    data_path = 'C:/Users/47450/Documents/ResQ Biometrics/Data sets/FEC_dataset/images/two-class_triplets'
    image_shape = (128, 128, 3)
    delta_trip_loss = 0.2
    embedding_size = 16 # faceNet uses 128, FECNet uses 16.
    batch_size = 8
    val_size = 0.05
    data_samples = 4000
    save_model_path = 'C:\\Users\\47450\\Documents\\ResQ Biometrics\\ResQ-Biometrics-CNN\\FECNet\\Models\\'
    new_model_name = 'FECNet_1_3.h5'
    load_weights = 'C:/Users/47450/Documents/ResQ Biometrics/ResQ-Biometrics-CNN/FECNet/Models/FECNet_1.h5'

    T = Train(data_path, save_model_path, new_model_name, image_shape, batch_size, delta_trip_loss,
                    embedding_size=16, N_data_samples=data_samples, model_type='FECNet',
                    callback_list=['tensorboard', 'save_best'], augment_data = True, optimizer = 'adam',
                    train_val_spilt=val_size, load_weights=load_weights, dir_or_mem='mem')

    T.fit()