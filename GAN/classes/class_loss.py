import tensorflow as tf


def g_loss(disc_generated_output, gen_output, target):
    LAMBDA = 100 #? test this
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


def l1loss(I_ae, I_ae_tilde):
    return tf.reduce_mean(tf.abs(I_ae - I_ae_tilde))

def softmax_cross_entropy(pred_fake, pred_truth, labels):
    truth = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=pred_truth)
    fake = tf.nn.softmax_cross_entropy_with_logits(label=labels, logits=pred_fake)
    softmax = truth + fake
    return softmax


def cGAN_loss(disc_real_output, disc_generated_output):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


def e_loss(label, pred):
    return tf.keras.losses.categorical_crossentropy(label, pred)
