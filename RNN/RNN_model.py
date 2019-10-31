from keras.applications import inception_v3, vgg16
from keras.layers import CuDNNLSTM
class RnnCnnModel:

    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = None

        
    """
    Choosing witch CNN network to use as the base model.
    """
    def add_base_network(self, base_model = 'inception', pree_trained = True, pooling = 'max'):
        if pree_trained:
            W = 'imagenet'
        else:
            W = None

        if base_model == 'inception':
            self.model = inception_v3.InceptionV3(include_top=False, weights=W, input_shape=self.input_shape, pooling=pooling)

        elif base_model == 'vgg16':
            self.model = vgg16.VGG16(include_top=False, weights=W, input_shape=self.input_shape, pooling=pooling)

        else:
            print('-- FROM SELF -- Choose from the list of base models')
            exit()
        

    def add_RNN(self):
        if not self.model:
            print('-- FROM SELF -- Add a base model first')
            exit()

        input_ = self.model.output
        

        
        



if __name__ == '__main__':
    input_shape = (128,128,3)
    model_class = RnnCnnModel(input_shape)
    model_class.add_base_network()