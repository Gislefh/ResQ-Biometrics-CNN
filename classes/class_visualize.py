from __future__ import print_function
import sys
sys.path.insert(0, 'C:\\Users\\47450\\Documents\\ResQ Biometrics\\ResQ-Biometrics-CNN\\classes')

from class_utils import test_model
from class_model import NonSecModel

import time
import numpy as np
from PIL import Image as pil_image
from keras.preprocessing.image import save_img
from keras import layers
from keras.applications import vgg16
from keras import backend as K
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.models import Model
from scipy.ndimage import zoom

import tensorflow as tf
import keras.backend.tensorflow_backend as ktf

## allow gpu growth
def get_session():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1,
                    allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
ktf.set_session(get_session())


class VisualizeLayers:

    def __init__(self, model, filter_range=(0, None), output_dim=(412, 412), upscaling_factor=1.2, upscaling_steps=9, step=1, epochs=15):

        self.model = model
        self.filter_range = filter_range
        self.output_dim = output_dim
        self.upscaling_steps = upscaling_steps
        self.upscaling_factor = upscaling_factor
        self.step = step
        self.epochs = epochs
        self.N_channels = self.model.input_shape[-1]
        self.images = []


    def visualize_layer(self, layer_name):

        assert len(self.model.inputs) == 1
        input_img = self.model.inputs[0]

        # get the symbolic outputs of each "key" layer (we gave them unique names).
        layer_dict = dict([(layer.name, layer) for layer in self.model.layers[1:]])

        output_layer = layer_dict[layer_name]
        assert isinstance(output_layer, layers.Conv2D)

        # Compute to be processed filter range
        filter_lower = self.filter_range[0]
        filter_upper = (self.filter_range[1]
                        if self.filter_range[1] is not None
                        else len(output_layer.get_weights()[1]))
        assert(filter_lower >= 0
            and filter_upper <= len(output_layer.get_weights()[1])
            and filter_upper > filter_lower)
        print('Compute filters {:} to {:}'.format(filter_lower, filter_upper))
     
        for f in range(filter_lower, filter_upper):
            print(f)
            img_loss = self.__generate_filter_image(input_img, output_layer.output, f)
            if img_loss is not None:
                self.images.append(img_loss[0])

                

    def __generate_filter_image(self, input_img,
                               layer_output,
                               filter_index):

        s_time = time.time()

        if K.image_data_format() == 'channels_first':
            loss = K.mean(layer_output[:, filter_index, :, :])
        else:
            loss = K.mean(layer_output[:, :, :, filter_index])
        
        
        # we compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient
        grads = self.__normalize(grads)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])
 
        # we start from a gray image with some random noise
        intermediate_dim = tuple(
            int(x / (self.upscaling_factor ** self.upscaling_steps)) for x in self.output_dim)
        if K.image_data_format() == 'channels_first':
            input_img_data = np.random.random(
                (1, self.N_channels, intermediate_dim[0], intermediate_dim[1])) 
        else:
            input_img_data = np.random.random(
                (1, intermediate_dim[0], intermediate_dim[1], self.N_channels)) 
        input_img_data = (input_img_data - 0.5) * 20 + 128


        # Slowly upscaling towards the original size prevents
        # a dominating high-frequency of the to visualized structure
        # as it would occur if we directly compute the 412d-image.
        # Behaves as a better starting point for each following dimension
        # and therefore avoids poor local minima
        for up in reversed(range(self.upscaling_steps)):
            # we run gradient ascent for e.g. 20 steps
            for _ in range(self.epochs):
                loss_value, grads_value = iterate([input_img_data])
                input_img_data += grads_value * self.step

                # some filters get stuck to 0, we can skip them
                #if loss_value <= K.epsilon():
                #    return None

            # Calculate upscaled dimension
            intermediate_dim = tuple(
                int(x / (self.upscaling_factor ** up)) for x in self.output_dim)
            # Upscale
            img = self.__deprocess_image(input_img_data[0])

            if self.N_channels == 3:
                img = np.array(pil_image.fromarray(img).resize(intermediate_dim, pil_image.BICUBIC))
            elif self.N_channels == 1:       
                img = zoom(img, (intermediate_dim[0] /np.shape(img)[0] , intermediate_dim[1]/np.shape(img)[1], 1), order = 1)
                input_img_data = np.expand_dims(input_img_data, -1)

            input_img_data = np.expand_dims(
                self.__process_image(img, input_img_data[0]), 0)
            
        # decode the resulting input image
        img = self.__deprocess_image(input_img_data[0])
        e_time = time.time()
        print('Costs of filter {:3}: {:5.0f} ( {:4.2f}s )'.format(filter_index,
                                                                  loss_value,
                                                                  e_time - s_time))
        return img, loss_value

    def get_filters(self):
        if self.images:
            return self.images
        else:
            print('No filter images generated. Generation filters')
            return None

    def __deprocess_image(self,x):

        # normalize tensor: center on 0., ensure std is 0.25
        x -= x.mean()
        x /= (x.std() + K.epsilon())
        x *= 0.25

        # clip to [0, 1]
        x += 0.5
        x = np.clip(x, 0, 1)

        # convert to RGB array
        x *= 255
        if K.image_data_format() == 'channels_first':
            x = x.transpose((1, 2, 0))
        x = np.clip(x, 0, 255).astype('uint8')
        return x


    def __process_image(self, x, former):
        if K.image_data_format() == 'channels_first':
            x = x.transpose((2, 0, 1))
        return (x / 255 - 0.5) * 4 * former.std() + former.mean()

    def __normalize(self, x):
        return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())


if __name__ == '__main__':
    my_model = True
    vgg_model = False

    if vgg_model:
        LAYER_NAME = 'block3_conv1'
        vgg = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        model = vgg
    
    elif my_model:    
        LAYER_NAME = 'conv2d_3'
        model1 = load_model('C:\\Users\\47450\\Documents\\ResQ Biometrics\\ResQ-Biometrics-CNN\\Models\\model_test_tensorboard_2.h5')
        end_layer_name = 'conv2d_4'
        model2= Model(inputs=model1.input, outputs=model1.get_layer(end_layer_name).output)
        model = model2

    else:
        exit()
    model.summary()


    VL = VisualizeLayers(model, step=1, epochs=10)#, output_dim=(200, 200), upscaling_steps=5, )
    VL.visualize_layer(LAYER_NAME)
    images = VL.get_filters()
    for i in images:
        plt.imshow(np.squeeze(i))
        plt.show()

   