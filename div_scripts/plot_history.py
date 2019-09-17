import numpy as np
import matplotlib.pyplot as plt

path_to_meta_data = 'C:\\Users\\47450\\Documents\\ResQ Biometrics\\ResQ-Biometrics-CNN\\Models\\From_Colab\\meta_data_model_16_09.npy'

meta_data = np.load(path_to_meta_data, allow_pickle = True)

history = meta_data.item()['model_history']

print(history['val_loss'])


epochs = list(range(len(history['val_loss'])))
val_loss = history['val_loss']
loss = history['loss']
val_acc = history['val_acc']
acc = history['acc']

plt.figure('Traning summary')
plt.subplot(211)
plt.plot(epochs, val_loss)
plt.plot(epochs, loss)
plt.legend(['val_loss', 'train_loss'])
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('loss')

plt.subplot(212)
plt.plot(epochs, val_acc)
plt.plot(epochs, acc)
plt.legend(['val_acc', 'train_acc'])
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('accuracy')

plt.show()

'''
meta_data = {'model_name' : model_name,
                'batch_size' : batch_size,
                'train_path' : train_path,
                'model_summary' : model.summary(),
                'model_classes': gen_train.get_classes(),
                'model_augmentations' : gen_train.get_aug(),
                'model_history' :  history,
                'model_input_shape' : X_shape,
}
'''