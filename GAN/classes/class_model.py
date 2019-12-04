""" IF-GAN structured models
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model


class GeneratorModels:

    def __init__(self, input_shape, model_type='Unet'):
        self.input_shape = input_shape
        # self.batch_size = batch_size # remove this sooon
        self.model_type = model_type

    def get_model(self):
        if self.model_type == 'Unet':
            model = self.__Unet_model()
        else:
            raise Exception('-- FROM SELF -- choose "Unet"')
        return model

    def __Unet_model(self):
        """
        TODO For this fuction to work the input shape has to be dividable by 2 all the way down. typ. input_shape[0] (and [1]) = 2^n
        TODO make it less hard-coded
        
        The input to the Unet is both images concatinated about the channel axis
        """

        def downsample(filters, size, apply_batchnorm=True):
            initializer = tf.random_normal_initializer(0., 0.02)

            result = tf.keras.Sequential()
            result.add(
                tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                       kernel_initializer=initializer, use_bias=False))

            if apply_batchnorm:
                result.add(tf.keras.layers.BatchNormalization())

            result.add(tf.keras.layers.LeakyReLU())

            return result

        def upsample(filters, size, apply_dropout=False):
            initializer = tf.random_normal_initializer(0., 0.02)

            result = tf.keras.Sequential()
            result.add(
                tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                                padding='same',
                                                kernel_initializer=initializer,
                                                use_bias=False))

            result.add(tf.keras.layers.BatchNormalization())

            if apply_dropout:
                result.add(tf.keras.layers.Dropout(0.5))

            result.add(tf.keras.layers.ReLU())

            return result

        # Init - include these in the function call. 
        init_N_filters = 32
        filter_size = 4

        inp = layers.Input(shape=(self.input_shape[0], self.input_shape[1], self.input_shape[2] * 2))
        down_layers_list = []
        current_first_dim = self.input_shape[0]
        init = True

        # Down-Sampling
        cnt = 0
        while current_first_dim >= filter_size:
            if init:
                x = downsample(init_N_filters, filter_size)(inp)
                init = False
            else:
                x = downsample(init_N_filters, filter_size)(x)
            down_layers_list.append(x)
            if cnt % 2 == 1:
                init_N_filters = init_N_filters * 2
            current_first_dim = self.input_shape[0] / np.power(2, len(down_layers_list))

            cnt += 1

        down_layers_list.reverse()
        init_N_filters = int(init_N_filters / 2)

        # Up-Sampling
        cnt = 0
        for i in range(len(down_layers_list) - 1):

            x = layers.Concatenate()([x, down_layers_list[i]])
            x = upsample(init_N_filters, filter_size)(x)
            if cnt % 2 == 1:
                init_N_filters = int(init_N_filters / 2)

            cnt += 1
        x = layers.Concatenate()([x, down_layers_list[-1]])
        output = upsample(3, filter_size)(x)

        model = Model(inp, output, name='Unet')
        return model

    """
    def __simple_MNIST_model(self): #from https://www.tensorflow.org/tutorials/generative/dcgan

        Model = tf.keras.Sequential()
        Model.add(layers.Dense(7*7*self.batch_size, use_bias=False, input_shape=(self.input_shape,)))
        Model.add(layers.BatchNormalization())
        Model.add(layers.LeakyReLU())

        Model.add(layers.Reshape((7, 7, self.batch_size)))
        #assert Model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

        Model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        #assert Model.output_shape == (None, 7, 7, 128)
        Model.add(layers.BatchNormalization())
        Model.add(layers.LeakyReLU())

        Model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        #assert Model.output_shape == (None, 14, 14, 64)
        Model.add(layers.BatchNormalization())
        Model.add(layers.LeakyReLU())

        Model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        #assert Model.output_shape == (None, 28, 28, 1)
        #Model.summary()
        #exit()
        return Model
    """


class DiscriminatorModel:

    def __init__(self, input_shape, model_type='patchGAN'):
        self.input_shape = input_shape
        self.model_type = model_type

    def get_model(self):
        if self.model_type == 'patchGAN':
            model = self.__patchGAN_model()
        else:
            raise Exception('-- FROM SELF -- choose "patchGAN"')
        return model

    def __patchGAN_model(self):
        def downsample(filters, size, apply_batchnorm=True):
            initializer = tf.random_normal_initializer(0., 0.02)

            result = tf.keras.Sequential()
            result.add(
                tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                       kernel_initializer=initializer, use_bias=False))

            if apply_batchnorm:
                result.add(tf.keras.layers.BatchNormalization())

            result.add(tf.keras.layers.LeakyReLU())

            return result

        initializer = tf.random_normal_initializer(0., 0.02)

        Ian = layers.Input(shape=self.input_shape, name='I_AN')
        Ise = layers.Input(shape=self.input_shape, name='I_SE')
        Iae = layers.Input(shape=self.input_shape, name='I_AE')

        x = layers.concatenate([Ian, Ise, Iae])

        down1 = downsample(64, 4, False)(x)
        down2 = downsample(128, 4)(down1)
        down3 = downsample(256, 4)(down2)

        zero_pad1 = layers.ZeroPadding2D()(down3)
        conv = layers.Conv2D(512, 4, strides=1,
                             kernel_initializer=initializer,
                             use_bias=False)(zero_pad1)

        batchnorm1 = layers.BatchNormalization()(conv)

        leaky_relu = layers.LeakyReLU()(batchnorm1)

        zero_pad2 = layers.ZeroPadding2D()(leaky_relu)

        last = layers.Conv2D(1, 4, strides=1,
                             kernel_initializer=initializer)(zero_pad2)

        model = Model(inputs=[Ian, Ise, Iae], outputs=last, name='PatchGAN')

        return model


class ClassificationModel:

    def __init__(self, input_shape, N_classes = 7, model_type='resNet' ):
        self.input_shape = input_shape
        self.model_type = model_type
        self.N_classes = N_classes

    def get_model(self):
        if self.model_type == 'resNet':
            self.model = self.__resNet_model()
        elif self.model_type == 'mobileNetv2':
            self.model = self.__mobileNetv2_model()
        else:
            raise Exception('-- FROM SELF -- choose either "resNet2 or "mobileNetv2.')
        return self.model

    def __resNet_model(self):
        start = layers.Input(shape = [self.input_shape[0], self.input_shape[1], self.input_shape[2]*2])
        onexone_conv = layers.Conv2D(3, (1,1))(start)

        resNet = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=self.input_shape,
                                                pooling='max')

        #input_layer = layers.Input(shape = self.input_shape)

        for layer in resNet.layers:
            if layer.name == 'add_9':
                break
            else:
                layer.trainable = False

        x = mobileNetv2.output(onexone_conv)
        x = layers.Dense(512)(mobileNetv2.output)
        output = layers.Dense(self.N_classes)(x)

        model = Model(inputs=resNet.input, outputs=output, name='ResNet50')
        return model

    def __mobileNetv2_model(self):
        start = layers.Input(shape = [self.input_shape[0], self.input_shape[1], self.input_shape[2]*2])
        onexone_conv = layers.Conv2D(3, (1,1))(start)
        #onexone_conv = layers.SeparableConv2D(3, (1,1))(start)
        mobileNetv2 = tf.keras.applications.MobileNetV2(input_shape=self.input_shape, alpha=1.0, include_top=False,
                                                        weights='imagenet', pooling='max')

        for layer in mobileNetv2.layers:
            if layer.name == 'block_8_add':
                break
            else:
                layer.trainable = False
        x = mobileNetv2(onexone_conv)
        x = layers.Dense(512)(mobileNetv2.output)
        output = layers.Dense(self.N_classes)(x)
        model = Model(inputs=mobileNetv2.input, outputs=output, name='MobileNetV2')

        return model


if __name__ == '__main__':
    input_shape = (256, 256, 3)
    batch_size = None
    GenModel = GeneratorModels(input_shape)
    gen_model = GenModel.get_model()
    DiscModel = DiscriminatorModel(input_shape)
    disc_model = DiscModel.get_model()
    ClassModel = ClassificationModel(input_shape, model_type='mobileNetv2')
    class_model = ClassModel.get_model()

    gen_model.summary()
    disc_model.summary()
    class_model.summary()

""" Simple model test
if __name__ == '__main__':
    #from https://www.tensorflow.org/tutorials/generative/dcgan
    from keras.datasets import mnist
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    BATCH_SIZE = 256
    INPUT_SHAPE_GEN = 100 
    BUFFER_SIZE = 60000
    noise_dim = 100
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    train_images = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]
    #train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE) # runs forever(isj) with this

    train_dataset = np.zeros((int(np.ceil(BUFFER_SIZE/BATCH_SIZE)), BATCH_SIZE, 28, 28, 1), dtype=np.float32)

    for i in range(int(np.ceil(BUFFER_SIZE/BATCH_SIZE))): # filled with zeros at the end - TODO fix
        im_block = train_images[i*BATCH_SIZE : i*BATCH_SIZE + BATCH_SIZE]
        train_dataset[i, 0:im_block.shape[0]] = im_block

    input_shape_disc = np.expand_dims(x_train[0], axis = -1).shape

    GenModels = GeneratorModels(INPUT_SHAPE_GEN, BATCH_SIZE)
    generator_model = GenModels.get_model()

    DiscModels = DiscriminatorModel(input_shape_disc, BATCH_SIZE)
    disc_model = DiscModels.get_model()


    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def discriminator_loss(real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    def train_step(images):
        noise = tf.random.normal([BATCH_SIZE, noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator_model(noise, training=True)

            real_output = disc_model(images, training=True)
            fake_output = disc_model(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator_model.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, disc_model.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator_model.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, disc_model.trainable_variables))

    def train(dataset, epochs):
        for epoch in range(epochs):

            print('epoch: ', epoch)
            for image_batch in tqdm(dataset):
                train_step(image_batch)
            
            for i in range(10):
                noise = np.expand_dims(np.random.rand(100) - 0.5, axis=0)
                pred = generator_model.predict(noise, batch_size=1)
                plt.figure(i)
                plt.imshow(pred[0, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.show()
            
                            

    train(train_dataset, epochs = 50)
"""
