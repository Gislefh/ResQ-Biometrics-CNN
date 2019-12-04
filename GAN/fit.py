import sys
import time
import numpy as np
import tensorflow as tf


sys.path.insert(0, "classes")
from class_model import GeneratorModels, DiscriminatorModel, ClassificationModel
from class_loss import e_loss, cGAN_loss, g_loss, l1loss, softmax_cross_entropy
from class_generator import DataGenerator
from utils import CreateAverages
## Constants
input_shape = (256, 256, 3)
batch_size = None

#loss function constants - epirically set from the paper
#lambda_1 = tf.constant([1]) 
#lambda_2 = tf.constant([200]) 
#lambda_3 = tf.constant([50]) 
lambda_1 = 1
lambda_2 = 200
lambda_3 = 50

## LOSS
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
expression_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

## Load Models
GenModel = GeneratorModels(input_shape)
g_model = GenModel.get_model()
DiscModel = DiscriminatorModel(input_shape)
d_model = DiscModel.get_model()
ClassModel = ClassificationModel(input_shape, model_type='mobileNetv2')
e_model = ClassModel.get_model()


## Data Generator
# I_ae and I_an
folder_with_landmarks = 'C:/Users/47450/Documents/ResQ Biometrics/Data sets/IF-GAN_averages/test/fear'
data_path_averages = 'C:/Users/47450/Documents/ResQ Biometrics/Data sets/ExpW/train_small'
CA = CreateAverages(data_path_averages)
I_ae = CA.get_iae()
I_an = CA.get_ian()


#@tf.function # <- ditch this?
def train_step(i_se, i_an, i_ae, labels):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.

    with tf.GradientTape(persistent=True) as disc_tape, tf.GradientTape(persistent=True) as gen_tape, tf.GradientTape(persistent=True) as exp_tape:

        """
        FEED-FORWARD
        """

        i_tilde = g_model(i_se, i_an)

        d_real = d_model(i_an, i_se, i_ae)
        d_fake = d_model(i_an, i_se, i_tilde)

        tilde_output = e_model(i_se, i_tilde)
        truth_output = e_model(i_se, i_ae)

        """
        LOSS - TODO test one loss for all models -vs- different loss function for the different models  
        """
        cGAN_loss_value = cGAN_loss(d_real, d_fake)
        L1_loss = l1loss(i_ae, i_tilde)
        
        tot_gan_loss = lambda_1 * cGAN_loss_value + lambda_2 * L1_loss
        loss_discriminator = tf.math.negative(tot_gan_loss) # negating the loss to make it maximize the value, as tf wants to minimize - i think. 
        loss_generator = tot_gan_loss

        # e_tilde_loss = e_loss(labels, tilde_output)
        # e_truth_loss = e_loss(labels, truth_output)
        e_loss = softmax_cross_entropy(tilde_output, truth_output, labels)

    """
    FIND GRADIENTS
    """
    generator_gradients = gen_tape.gradient(loss_generator, g_model.trainable_variables)

    discriminator_gradients = disc_tape.gradient(loss_discriminator, d_model.trainable_variables)
    
    expression_gradients = exp_tape.gradient(e_loss, e_model.trainable_variables)

    """
    BACKPROB WITH GRADIENTS
    """
    generator_optimizer.apply_gradients(zip(generator_gradients,
                                          g_model.trainable_variables))

    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              d_model.trainable_variables))

    expression_optimizer.apply_gradients(zip(expression_gradients,
                                              e_model.trainable_variables))





I_ae_to_trainstep = np.zeros(batch_size, input_shape[0], input_shape[1], input_shape[2])

for epoch in range(EPOCHS):

    for I_se, labels in data_gen:
        label_one_hot = one_hot(labels)
        for i in range(len(labels)):
            I_ae_to_trainstep[i] = i_ae[labels[i]]

        train_step(i_se, i_an, I_ae_to_trainstep, label_one_hot)
        
        
        
        
        
        #if n % 10 == 0:
        #    print('.', end='')
        #n += 1

    #clear_output(wait=True)
    # Using a consistent image (sample_horse) so that the progress of the model
    # is clearly visible.
    #generate_images(generator_g, sample_horse)

    #if (epoch + 1) % 5 == 0:
    #    ckpt_save_path = ckpt_manager.save()
    #    print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
    #                                                        ckpt_save_path))

    #print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
    #                                                   time.time() - start))
