import tensorflow as tf
import numpy as np
#-------------------------------------
def NN2_model():
    import keras
    from keras.models import Sequential, Model
    from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
    from keras.layers.core import Dense, Activation, Lambda, Flatten
    from keras.layers.normalization import BatchNormalization
    from keras import backend as K
    from keras.layers.pooling import MaxPooling2D, AveragePooling2D
    from keras.layers.merge import Concatenate

    myInput = Input(shape=(96, 96, 3))

    x = ZeroPadding2D(padding=(3, 3), input_shape=(96, 96, 3))(myInput)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=3, epsilon=0.00001, name='bn1')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D(pool_size=3, strides=2)(x)
    x = Lambda(lambda x: tf.nn.lrn(x, alpha=1e-4, beta=0.75), name='lrn_1')(x)
    x = Conv2D(64, (1, 1), name='conv2')(x)
    x = BatchNormalization(axis=3, epsilon=0.00001, name='bn2')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(192, (3, 3), name='conv3')(x)
    x = BatchNormalization(axis=3, epsilon=0.00001, name='bn3')(x)
    x = Activation('relu')(x)
    Lambda(lambda x: tf.nn.lrn(x, alpha=1e-4, beta=0.75), name='lrn_2')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D(pool_size=3, strides=2)(x)

    # Inception3a
    inception_3a_3x3 = Conv2D(96, (1, 1), name='inception_3a_3x3_conv1')(x)
    inception_3a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_3x3_bn1')(inception_3a_3x3)
    inception_3a_3x3 = Activation('relu')(inception_3a_3x3)
    inception_3a_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3a_3x3)
    inception_3a_3x3 = Conv2D(128, (3, 3), name='inception_3a_3x3_conv2')(inception_3a_3x3)
    inception_3a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_3x3_bn2')(inception_3a_3x3)
    inception_3a_3x3 = Activation('relu')(inception_3a_3x3)

    inception_3a_5x5 = Conv2D(16, (1, 1), name='inception_3a_5x5_conv1')(x)
    inception_3a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_5x5_bn1')(inception_3a_5x5)
    inception_3a_5x5 = Activation('relu')(inception_3a_5x5)
    inception_3a_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3a_5x5)
    inception_3a_5x5 = Conv2D(32, (5, 5), name='inception_3a_5x5_conv2')(inception_3a_5x5)
    inception_3a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_5x5_bn2')(inception_3a_5x5)
    inception_3a_5x5 = Activation('relu')(inception_3a_5x5)

    inception_3a_pool = MaxPooling2D(pool_size=3, strides=2)(x)
    inception_3a_pool = Conv2D(32, (1, 1), name='inception_3a_pool_conv')(inception_3a_pool)
    inception_3a_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_pool_bn')(inception_3a_pool)
    inception_3a_pool = Activation('relu')(inception_3a_pool)
    inception_3a_pool = ZeroPadding2D(padding=((3, 4), (3, 4)))(inception_3a_pool)

    inception_3a_1x1 = Conv2D(64, (1, 1), name='inception_3a_1x1_conv')(x)
    inception_3a_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_1x1_bn')(inception_3a_1x1)
    inception_3a_1x1 = Activation('relu')(inception_3a_1x1)

    inception_3a = concatenate([inception_3a_3x3, inception_3a_5x5, inception_3a_pool, inception_3a_1x1], axis=3)

    # Inception3b
    inception_3b_3x3 = Conv2D(96, (1, 1), name='inception_3b_3x3_conv1')(inception_3a)
    inception_3b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_3x3_bn1')(inception_3b_3x3)
    inception_3b_3x3 = Activation('relu')(inception_3b_3x3)
    inception_3b_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3b_3x3)
    inception_3b_3x3 = Conv2D(128, (3, 3), name='inception_3b_3x3_conv2')(inception_3b_3x3)
    inception_3b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_3x3_bn2')(inception_3b_3x3)
    inception_3b_3x3 = Activation('relu')(inception_3b_3x3)

    inception_3b_5x5 = Conv2D(32, (1, 1), name='inception_3b_5x5_conv1')(inception_3a)
    inception_3b_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_5x5_bn1')(inception_3b_5x5)
    inception_3b_5x5 = Activation('relu')(inception_3b_5x5)
    inception_3b_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3b_5x5)
    inception_3b_5x5 = Conv2D(64, (5, 5), name='inception_3b_5x5_conv2')(inception_3b_5x5)
    inception_3b_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_5x5_bn2')(inception_3b_5x5)
    inception_3b_5x5 = Activation('relu')(inception_3b_5x5)

    inception_3b_pool = Lambda(lambda x: x**2, name='power2_3b')(inception_3a)
    inception_3b_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_3b_pool)
    inception_3b_pool = Lambda(lambda x: x*9, name='mult9_3b')(inception_3b_pool)
    inception_3b_pool = Lambda(lambda x: K.sqrt(x), name='sqrt_3b')(inception_3b_pool)
    inception_3b_pool = Conv2D(64, (1, 1), name='inception_3b_pool_conv')(inception_3b_pool)
    inception_3b_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_pool_bn')(inception_3b_pool)
    inception_3b_pool = Activation('relu')(inception_3b_pool)
    inception_3b_pool = ZeroPadding2D(padding=(4, 4))(inception_3b_pool)

    inception_3b_1x1 = Conv2D(64, (1, 1), name='inception_3b_1x1_conv')(inception_3a)
    inception_3b_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_1x1_bn')(inception_3b_1x1)
    inception_3b_1x1 = Activation('relu')(inception_3b_1x1)

    inception_3b = concatenate([inception_3b_3x3, inception_3b_5x5, inception_3b_pool, inception_3b_1x1], axis=3)

    # Inception3c
    inception_3c_3x3 = Conv2D(128, (1, 1), strides=(1, 1), name='inception_3c_3x3_conv1')(inception_3b)
    inception_3c_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3c_3x3_bn1')(inception_3c_3x3)
    inception_3c_3x3 = Activation('relu')(inception_3c_3x3)
    inception_3c_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3c_3x3)
    inception_3c_3x3 = Conv2D(256, (3, 3), strides=(2, 2), name='inception_3c_3x3_conv'+'2')(inception_3c_3x3)
    inception_3c_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3c_3x3_bn'+'2')(inception_3c_3x3)
    inception_3c_3x3 = Activation('relu')(inception_3c_3x3)

    inception_3c_5x5 = Conv2D(32, (1, 1), strides=(1, 1), name='inception_3c_5x5_conv1')(inception_3b)
    inception_3c_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3c_5x5_bn1')(inception_3c_5x5)
    inception_3c_5x5 = Activation('relu')(inception_3c_5x5)
    inception_3c_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3c_5x5)
    inception_3c_5x5 = Conv2D(64, (5, 5), strides=(2, 2), name='inception_3c_5x5_conv'+'2')(inception_3c_5x5)
    inception_3c_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3c_5x5_bn'+'2')(inception_3c_5x5)
    inception_3c_5x5 = Activation('relu')(inception_3c_5x5)

    inception_3c_pool = MaxPooling2D(pool_size=3, strides=2)(inception_3b)
    inception_3c_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(inception_3c_pool)

    inception_3c = concatenate([inception_3c_3x3, inception_3c_5x5, inception_3c_pool], axis=3)

    #inception 4a
    inception_4a_3x3 = Conv2D(96, (1, 1), strides=(1, 1), name='inception_4a_3x3_conv'+'1')(inception_3c)
    inception_4a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4a_3x3_bn'+'1')(inception_4a_3x3)
    inception_4a_3x3 = Activation('relu')(inception_4a_3x3)
    inception_4a_3x3 = ZeroPadding2D(padding=(1, 1))(inception_4a_3x3)
    inception_4a_3x3 = Conv2D(192, (3, 3), strides=(1, 1), name='inception_4a_3x3_conv'+'2')(inception_4a_3x3)
    inception_4a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4a_3x3_bn'+'2')(inception_4a_3x3)
    inception_4a_3x3 = Activation('relu')(inception_4a_3x3)

    inception_4a_5x5 = Conv2D(32, (1,1), strides=(1,1), name='inception_4a_5x5_conv1')(inception_3c)
    inception_4a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4a_5x5_bn1')(inception_4a_5x5)
    inception_4a_5x5 = Activation('relu')(inception_4a_5x5)
    inception_4a_5x5 = ZeroPadding2D(padding=(2,2))(inception_4a_5x5)
    inception_4a_5x5 = Conv2D(64, (5,5), strides=(1,1), name='inception_4a_5x5_conv'+'2')(inception_4a_5x5)
    inception_4a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4a_5x5_bn'+'2')(inception_4a_5x5)
    inception_4a_5x5 = Activation('relu')(inception_4a_5x5)

    inception_4a_pool = Lambda(lambda x: x**2, name='power2_4a')(inception_3c)
    inception_4a_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_4a_pool)
    inception_4a_pool = Lambda(lambda x: x*9, name='mult9_4a')(inception_4a_pool)
    inception_4a_pool = Lambda(lambda x: K.sqrt(x), name='sqrt_4a')(inception_4a_pool)

    inception_4a_pool = Conv2D(128, (1,1), strides=(1,1), name='inception_4a_pool_conv'+'')(inception_4a_pool)
    inception_4a_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4a_pool_bn'+'')(inception_4a_pool)
    inception_4a_pool = Activation('relu')(inception_4a_pool)
    inception_4a_pool = ZeroPadding2D(padding=(2, 2))(inception_4a_pool)

    inception_4a_1x1 = Conv2D(256, (1, 1), strides=(1, 1), name='inception_4a_1x1_conv'+'')(inception_3c)
    inception_4a_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4a_1x1_bn'+'')(inception_4a_1x1)
    inception_4a_1x1 = Activation('relu')(inception_4a_1x1)

    inception_4a = concatenate([inception_4a_3x3, inception_4a_5x5, inception_4a_pool, inception_4a_1x1], axis=3)

    #inception4e
    inception_4e_3x3 = Conv2D(160, (1,1), strides=(1,1), name='inception_4e_3x3_conv'+'1')(inception_4a)
    inception_4e_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4e_3x3_bn'+'1')(inception_4e_3x3)
    inception_4e_3x3 = Activation('relu')(inception_4e_3x3)
    inception_4e_3x3 = ZeroPadding2D(padding=(1, 1))(inception_4e_3x3)
    inception_4e_3x3 = Conv2D(256, (3,3), strides=(2,2), name='inception_4e_3x3_conv'+'2')(inception_4e_3x3)
    inception_4e_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4e_3x3_bn'+'2')(inception_4e_3x3)
    inception_4e_3x3 = Activation('relu')(inception_4e_3x3)

    inception_4e_5x5 = Conv2D(64, (1,1), strides=(1,1), name='inception_4e_5x5_conv'+'1')(inception_4a)
    inception_4e_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4e_5x5_bn'+'1')(inception_4e_5x5)
    inception_4e_5x5 = Activation('relu')(inception_4e_5x5)
    inception_4e_5x5 = ZeroPadding2D(padding=(2, 2))(inception_4e_5x5)
    inception_4e_5x5 = Conv2D(128, (5,5), strides=(2,2), name='inception_4e_5x5_conv'+'2')(inception_4e_5x5)
    inception_4e_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_4e_5x5_bn'+'2')(inception_4e_5x5)
    inception_4e_5x5 = Activation('relu')(inception_4e_5x5)

    inception_4e_pool = MaxPooling2D(pool_size=3, strides=2)(inception_4a)
    inception_4e_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(inception_4e_pool)

    inception_4e = concatenate([inception_4e_3x3, inception_4e_5x5, inception_4e_pool], axis=3)
    # --------------------
    # to dense-net
    #TODO - DenseNet here 


    # --------------------
    #fc -layers
    out_shape = inception_4e.shape[1:3]
    x = AveragePooling2D(pool_size = out_shape)(inception_4e)
    x = Dense(512, activation='reluy')

    # --------------------
    #L2 norm



    # --------------------
    # embedding



    #model = Model(inputs=[myInput], outputs=inception_4e)
    #model.summary()

    #return model

"""
-- Triplet loss function -- 
gt = 1 or 2 or 3:
    for gt = 1 => e1 and e2 is the most similar
    for gt = 2 => e2 and e3 is the moset similar    ### assuming this is correct - check that
    for gt = 3 => e1 and e3 is the moset similar    ### assuming this is correct - check that

e_{1,2,3} - embedding

delta - a small margin
"""
def trip_loss(e1, e2, e3, gt, delta = 0.1):
        
    if gt == 1:
        dist1 = distance.euclidean(e1, e2)**2 - distance.euclidean(e1, e3)**2 + delta
        dist2 = distance.euclidean(e1, e2)**2 - distance.euclidean(e2, e3)**2 + delta
        loss = np.maximum(0, dist1) + np.maximum(0, dist2)

    elif gt == 2:
        dist1 = distance.euclidean(e2, e3)**2 - distance.euclidean(e1, e3)**2 + delta
        dist2 = distance.euclidean(e2, e3)**2 - distance.euclidean(e1, e2)**2 + delta
        loss = np.maximum(0, dist1) + np.maximum(0, dist2)
    
    elif gt == 3:
        dist1 = distance.euclidean(e1, e3)**2 - distance.euclidean(e2, e3)**2 + delta
        dist2 = distance.euclidean(e1, e3)**2 - distance.euclidean(e1, e2)**2 + delta
        loss = np.maximum(0, dist1) + np.maximum(0, dist2)

    else:
        print('--- Ground Truth not in the scope ---')
        return None
    
    return loss




from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input, GlobalMaxPooling2D, Embedding, Lambda, concatenate
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

    #mixed8 is the end of the inception 4e block - used in FECNet
    for i, layer in enumerate(incV3.layers):
        if layer.name == 'mixed8':
            layer_4e_index = i
            break

    # insert denseNet here

    input_ = incV3.input
    x = incV3.layers[layer_4e_index].output
    out = Dense(embedding_size)(x)
    image_embedder = Model(input_, out)
    image_embedder.summary()
    exit()

    input_a = Input((input_shape[0], input_shape[1], input_shape[2]), name='anchor')
    input_p = Input((input_shape[0], input_shape[1], input_shape[2]), name='positive')
    input_n = Input((input_shape[0], input_shape[1], input_shape[2]), name='negative')

    normalize = Lambda(lambda x: l2_normalize(x, axis=-1), name='normalize')

    x = image_embedder(input_a)
    output_a = normalize(x)
    x = image_embedder(input_p)
    output_p = normalize(x)
    x = image_embedder(input_n)
    output_n = normalize(x)

    merged_vector = concatenate([output_a, output_p, output_n], axis=-1)

    model = Model(inputs=[input_a, input_p, input_n],
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