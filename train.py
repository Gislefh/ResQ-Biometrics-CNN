from datetime import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sb

from classes.class_generator import Generator
from classes.class_utils import get_inception_w_imnet

train_path = '../Data sets/kmeans_small/'
new_model_name = 'classifier_cluster_0'
save_model_path = 'models\\'

if new_model_name in os.listdir(save_model_path):
    print('Model name exists. Change the model name')

# consts
N_channels = 3
N_images_per_class = 4000
batch_size = 150
image_shape = (128, 128)
N_classes = 15
X_shape = (batch_size, image_shape[0], image_shape[1], N_channels)
Y_shape = (batch_size, N_classes)
val_size = 0.2

gen_train = Generator(train_path,
                      X_shape,
                      Y_shape,
                      N_classes,
                      N_channels,
                      batch_size,
                      train_val_split=val_size,
                      N_images_per_class=N_images_per_class)

gen_train.add_noise(0.1)
gen_train.add_rotate(max_abs_angle_deg=30)
gen_train.add_gamma_transform(0.5, 1.5)
gen_train.add_flip()

train_gen = gen_train.flow_from_dir(set='train', crop=False)
val_gen = gen_train.flow_from_dir(set='val', augment_validation=False, crop=False)

# for x, y in train_gen:
#     plt.imshow(x[0])
#     plt.show()

model = get_inception_w_imnet(input_shape=(image_shape[0], image_shape[1], N_channels), N_classes=N_classes,
                              freeze_layers=True)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              min_delta=0,
                                              patience=10,
                                              verbose=0,
                                              mode='auto',
                                              baseline=None,
                                              restore_best_weights=True)

save_best = tf.keras.callbacks.ModelCheckpoint(save_model_path + new_model_name,
                                               monitor='val_loss',
                                               verbose=1,
                                               save_best_only=True,
                                               save_weights_only=False,
                                               mode='min',
                                               save_freq='epoch')

tensorboard_name = datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard = tf.keras.callbacks.TensorBoard(log_dir=save_model_path + 'logs\\' + tensorboard_name,
                                             histogram_freq=0,
                                             write_graph=True,
                                             write_images=True,
                                             embeddings_freq=0,
                                             embeddings_layer_names=None,
                                             embeddings_metadata=None,
                                             embeddings_data=None,
                                             update_freq='batch')

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

steps_per_epoch = int(np.floor(gen_train.get_length_data() * (1 - val_size)) / batch_size)
val_steps_per_epoch = int(np.floor(gen_train.get_length_data() * val_size) / batch_size)

best_loss = 10000


def show_confusion_matrix():
    global cnt, x, y
    conf_matrix = np.zeros((N_classes, N_classes))
    cnt = 0
    for x, y in val_gen:
        for i in range(x.shape[0]):
            pred = model.predict(np.expand_dims(x[i], 0))
            pred = np.argmax(pred)
            gt = np.argmax(y[i])
            conf_matrix[pred, gt] = conf_matrix[pred, gt] + 1
            cnt += 1
        print(cnt, '/', val_steps_per_epoch * batch_size)
        
        if cnt > val_steps_per_epoch * batch_size:
            break
    df = pd.DataFrame(conf_matrix, index=os.listdir(train_path), columns=os.listdir(train_path))
    plt.figure()
    sb.heatmap(df, annot=True)
    plt.show()


def train_model():
    global cnt, metrics_names, best_loss
    for epoch in range(20):
        model.reset_metrics()
        
        for cnt, (image_batch, label_batch) in enumerate(train_gen):
            result = model.train_on_batch(image_batch, label_batch, reset_metrics=False)
            metrics_names = model.metrics_names
            print("batch: ",
                  "{}/{}  ".format(cnt, steps_per_epoch),
                  "train: ",
                  "{}: {:.3f}".format(metrics_names[0], result[0]),
                  "{}: {:.3f}".format(metrics_names[1], result[1]))
            
            if cnt >= steps_per_epoch:
                break
        
        for cnt, (image_batch, label_batch) in enumerate(val_gen):
            result = model.test_on_batch(image_batch, label_batch,
                                         reset_metrics=False)
            
            if cnt >= val_steps_per_epoch:
                break
        
        metrics_names = model.metrics_names
        print("\neval: ",
              "{}: {:.3f}".format(metrics_names[0], result[0]),
              "{}: {:.3f}".format(metrics_names[1], result[1]))
        
        if best_loss > result[0]:
            best_loss = result[0]
            model.save(save_model_path + '/' + new_model_name + str(epoch), overwrite=False, save_format='h5')
        
        show_confusion_matrix()


train_model()


