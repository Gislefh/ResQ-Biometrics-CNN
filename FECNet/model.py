import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input, GlobalMaxPooling2D, Embedding, Lambda, concatenate, BatchNormalization, Activation
from tensorflow.keras.models import Sequential, Model
from tensorflow.math import l2_normalize
from tensorflow.compat.v2.keras.applications.inception_v3 import InceptionV3
from scipy.spatial import distance

def tensorflow_model(input_shape):
    # input 
    input_layer = Input(shape=input_shape)

    # block 1
    x = Conv2D(32, 3, activation='relu')(input_layer)
    x = Conv2D(32, 3, activation='relu')(x)
    x = Conv2D(32, 3, activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.1)(x)

    # block 2
    x = Conv2D(64, 3, activation='relu')(x)
    x = Conv2D(64, 3, activation='relu')(x)
    x = Conv2D(64, 3, activation='relu')(x)
    
    # Global pooling
    x = GlobalMaxPooling2D()(x)
    x = Dropout(0.1)(x)

    # L2 normalization
    x = l2_normalize(x)

    # Embedding
    #x = Embedding(input_dim=x.shape, output_dim=128)
    x = Dense(128)(x)

    model = Model(inputs=input_layer, outputs=x)

    return model


def faceNet_inceptionv3_model(input_shape, embedding_size):
    # import inceptionV3. TODO - remove last n layers to get the otput of the model to be block 4e 
    incV3 = InceptionV3(include_top=False, weights='imagenet', input_shape=input_shape, pooling='max')

    #mixed8 is the end of the inception 4e block - used in FECNet
    for i, layer in enumerate(incV3.layers):
        if layer.name == 'mixed8':
            layer_4e_index = i
            break
        
    # use last layer
    layer_4e_index = -1

    input_ = incV3.input
    x = incV3.layers[layer_4e_index].output
    out = Dense(embedding_size)(x)

    image_embedder = Model(input_, out)



    input_1 = Input((input_shape[0], input_shape[1], input_shape[2]), name='image_1')
    input_2 = Input((input_shape[0], input_shape[1], input_shape[2]), name='image_2')
    input_3 = Input((input_shape[0], input_shape[1], input_shape[2]), name='image_3')

    normalize = Lambda(lambda x: l2_normalize(x, axis=-1), name='normalize')

    x = image_embedder(input_1)
    output_1 = normalize(x)
    x = image_embedder(input_2)
    output_2 = normalize(x)
    x = image_embedder(input_3)
    output_3 = normalize(x)

    merged_vector = concatenate([output_1, output_2, output_3], axis=-1)

    model = Model(inputs=[input_1, input_2, input_3],
                  outputs=merged_vector)
    return model


def FECNet_inceptionv3_model(input_shape, embedding_size):
    # import inceptionV3. TODO - remove last n layers to get the otput of the model to be block 4e 
    incV3 = InceptionV3(include_top=False, weights='imagenet', input_shape=input_shape, pooling='max')

    # mixed8 is the end of the inception 4e block - used in FECNet
    for i, layer in enumerate(incV3.layers):
        if layer.name == 'mixed8':
            layer_4e_index = i
            break
    
    # freeze the first n layers
    for i, layer in enumerate(incV3.layers):
        layer.trainable = False
        if layer.name == 'mixed6':
            break

    input_ = incV3.input
    x = incV3.layers[layer_4e_index].output
    
    #TODO insert denseNet here and maybe freeze some layers at the bottom if inception

    x = GlobalMaxPooling2D()(x)
    x = Dense(512)(x)
    out = Dense(embedding_size)(x)
    image_embedder = Model(input_, out)

    image_1 = Input((input_shape[0], input_shape[1], input_shape[2]), name='im1')
    image_2 = Input((input_shape[0], input_shape[1], input_shape[2]), name='im2')
    image_3 = Input((input_shape[0], input_shape[1], input_shape[2]), name='im3')

    normalize = Lambda(lambda x: l2_normalize(x, axis=-1), name='normalize')

    x = image_embedder(image_1)
    output_1 = normalize(x)
    x = image_embedder(image_2)
    output_2 = normalize(x)
    x = image_embedder(image_3)
    output_3 = normalize(x)

    merged_vector = concatenate([output_1, output_2, output_3], axis=-1)

    model = Model(inputs=[image_1, image_2, image_3],
                  outputs=merged_vector)

    return model

def FECNet_inceptionv3_dense_model(input_shape, embedding_size):
    # Inceptionv3
    incV3 = InceptionV3(include_top=False, weights='imagenet', input_shape=input_shape, pooling='max')

    # Freeze the first n layers
    for i, layer in enumerate(incV3.layers):
        layer.trainable = False
        if layer.name == 'mixed6':
            break

    # mixed8 is the end of the inception 4e block - used in FECNet
    for i, layer in enumerate(incV3.layers):
        if layer.name == 'mixed8':
            layer_4e_index = i
            break
    
    #1x1 conv layer with 512 filters
    x = Conv2D(512, 1)(incV3.layers[layer_4e_index].output)

    # Dense block with 5 layers and groth rate of 64
    def convolution_block(x, nb_channels, dropout_rate=None, bottleneck=False, weight_decay=1e-4):
        """
        Creates a convolution block consisting of BN-ReLU-Conv.
        Optional: bottleneck, dropout
        """

        # Bottleneck
        if bottleneck:
            bottleneckWidth = 4
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(nb_channels * bottleneckWidth, (1, 1),
                                        kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
            # Dropout
            if dropout_rate:
                x = Dropout(dropout_rate)(x)

        # Standard (BN-ReLU-Conv)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(nb_channels, (3, 3), padding='same')(x)

        # Dropout
        if dropout_rate:
            x = Dropout(dropout_rate)(x)

        return x
    def dense_block(x, nb_layers, nb_channels, growth_rate, dropout_rate=None, bottleneck=False,
                    weight_decay=1e-4):
        """
        Creates a dense block and concatenates inputs
        """
        for i in range(nb_layers):
            cb = convolution_block(x, growth_rate, dropout_rate, bottleneck)
            nb_channels += growth_rate
            x = concatenate([cb, x])
        return x, nb_channels

    growth_rate = 64
    nb_layers = 5
    x, nb_channels = dense_block(x, nb_layers, nb_layers, growth_rate, dropout_rate=None, bottleneck=False,
                    weight_decay=1e-4)


    input_ = incV3.input
    x = GlobalMaxPooling2D()(x)
    x = Dense(512)(x)
    out = Dense(embedding_size)(x)
    image_embedder = Model(input_, out)

 

    image_1 = Input((input_shape[0], input_shape[1], input_shape[2]), name='im1')
    image_2 = Input((input_shape[0], input_shape[1], input_shape[2]), name='im2')
    image_3 = Input((input_shape[0], input_shape[1], input_shape[2]), name='im3')

    normalize = Lambda(lambda x: l2_normalize(x, axis=-1), name='normalize')

    x = image_embedder(image_1)
    output_1 = normalize(x)
    x = image_embedder(image_2)
    output_2 = normalize(x)
    x = image_embedder(image_3)
    output_3 = normalize(x)

    merged_vector = concatenate([output_1, output_2, output_3], axis=-1)

    model = Model(inputs=[image_1, image_2, image_3],
                  outputs=merged_vector)

    return model

def test_siam_model(input_shape, embedding_size):

    def base_net(input_shape):
        input_ = Input(shape=input_shape)
        x = Flatten()(input_)
        x = Dense(50, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(50, activation='relu')(x)
        return Model(input_, x)
    
    base_model = base_net(input_shape = input_shape)

    input_1 = Input(shape=input_shape)
    input_2 = Input(shape=input_shape)
    input_3 = Input(shape=input_shape)

    processed_1 = base_model(input_1)
    processed_2 = base_model(input_2)
    processed_3 = base_model(input_3)

    normalize = Lambda(lambda x: l2_normalize(x, axis=-1), name='normalize')

    x = normalize(processed_1)
    out_1 = Dense(embedding_size)(x)

    x = normalize(processed_2)
    out_2 = Dense(embedding_size)(x)

    x = normalize(processed_3)
    out_3 = Dense(embedding_size)(x)

    merged_vector = concatenate([out_1, out_2, out_3], axis=-1)

    # triplet loss layer here, maybe


    model = Model(inputs=[input_1, input_2, input_3], outputs=merged_vector)
    return model

if __name__ == "__main__":
    import os 
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"   
    model = FECNet_inceptionv3_dense_model((224, 224, 3), 24)
    model.summary()
    exit()
    model = test_siam_model((100, 100, 3), 24)
    model.summary()
    
    from keras.utils import plot_model
    plot_model(model, to_file='model.png')

    """
    from keras import backend as K
    from keras.layers import Layer

    class MyLayer(Layer):

        def __init__(self, output_dim, **kwargs):
            self.output_dim = output_dim
            super(MyLayer, self).__init__(**kwargs)

        def build(self, input_shape):
            # Create a trainable weight variable for this layer.
            self.kernel = self.add_weight(name='kernel', 
                                        shape=(input_shape[1], self.output_dim),
                                        initializer='uniform',
                                        trainable=True)
            super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

        def call(self, x):
            return K.dot(x, self.kernel)

        def compute_output_shape(self, input_shape):
            return (input_shape[0], self.output_dim)
    """