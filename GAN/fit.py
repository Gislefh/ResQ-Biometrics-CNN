import sys
import time
import numpy as np

from GAN.classes.class_model import GeneratorModels, DiscriminatorModel, ClassificationModel

sys.path.insert(0, "classes")

import tensorflow as tf
from class_model import *
from class_loss import e_loss, d_loss, g_loss


# Get the predictions.
# Calculate the loss.
# Calculate the gradients using backpropagation.
# Apply the gradients to the optimizer.


@tf.function
def train_step(i_se, i_an, i_ae, labels):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.

    real = 1
    fake = 0

    with tf.GradientTape(persistent=True) as tape:

        """
        OUTPUT
        """

        i_tilde = g_model(i_se, i_an)

        d_real = d_model(i_an, i_se, i_ae)
        d_fake = d_model(i_an, i_se, i_tilde)

        tilde_output = e_model(i_se, i_tilde)
        truth_output = e_model(i_se, i_ae)

        """
        LOSS
        """
        disc_loss = d_loss(d_real, d_fake)
        gen_loss = g_loss(disc_loss, i_ae, i_tilde)

        e_tilde_loss = e_loss(labels, tilde_output)
        e_truth_loss = e_loss(labels, truth_output)




    """
    INSERT GRADIENTS
    """


    """
    INSERT OPTIMIZER
    """



input_shape = (256, 256, 3)
batch_size = None
GenModel = GeneratorModels(input_shape)
g_model = GenModel.get_model()
DiscModel = DiscriminatorModel(input_shape)
d_model = DiscModel.get_model()
ClassModel = ClassificationModel(input_shape, model_type='mobileNetv2')
e_model = ClassModel.get_model()

path_to_average_neutral = ""

i_an = path_to_average_neutral # get the average image, not correct

i_ae = np.array(["list","of","class","averages"])


for epoch in range(EPOCHS):
    start = time.time()

    n = 0
    for x,y in [image, label]:
        train_step(x, i_an, i_ae, y)
        if n % 10 == 0:
            print('.', end='')
        n += 1

    clear_output(wait=True)
    # Using a consistent image (sample_horse) so that the progress of the model
    # is clearly visible.
    generate_images(generator_g, sample_horse)

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                            ckpt_save_path))

    print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                       time.time() - start))
