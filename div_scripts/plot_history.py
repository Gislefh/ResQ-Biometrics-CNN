import numpy as np
import matplotlib.pyplot as plt

path_to_meta_data = 'C:\\Users\\47450\\Documents\\ResQ Biometrics\\ResQ-Biometrics-CNN\\Models\\From_Colab\\meta_data_model_4.npy'

meta_data = np.load(path_to_meta_data, allow_pickle = True)

history = meta_data.item()['model_history']

print(history.history)







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