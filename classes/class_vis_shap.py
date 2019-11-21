import shap
from keras.applications.vgg16 import preprocess_input
import keras.backend as K
import numpy as np

class VisualizeShap:


    def __init__(self, model, label_list):
        self.model = model
        self.label_list = label_list


    def explain_layer(self, to_explain, layer_nr = 1):
        def map2layer(x, layer):
            feed_dict = dict(zip([self.model.layers[0].input], [preprocess_input(x.copy())]))
            return K.get_session().run(self.model.layers[layer].input, feed_dict)

        e = shap.GradientExplainer(
            (self.model.layers[7].input, self.model.layers[-1].output),
            map2layer(X, 7),
            local_smoothing=0 # std dev of smoothing noise
            )

        shap_values,indexes = e.shap_values(map2layer(to_explain, layer_nr), ranked_outputs=2)

        # get the names for the classes
        index_names = np.vectorize(lambda x: self.label_list[str(x)][1])(indexes)

        # plot the explanations
        shap.image_plot(shap_values, to_explain, index_names)


if __name__ == '__main__':
    from keras.models import load_model
    from class_generator import Generator



    ### data
    data_path = 'C:/Users/47450/Documents/ResQ Biometrics/Data sets/ExpW/train'
    label_list = ['angry', 'happy', 'neutral']
    batch_size = 1
    X_shape = (batch_size, 100, 100, 3)
    Y_shape = (batch_size, len(label_list))
    N_channels = X_shape[-1]
    N_classes = len(label_list)
    
    G = Generator(data_path, X_shape, Y_shape, N_classes, N_channels, batch_size, train_val_split = 0.0, class_list = label_list, N_images_per_class = None)
    gen = G.flow_from_dir(set = 'test')

    for x, y in gen:
        X = x
        break

    ### model
    model_path = 'C:/Users/47450/Documents/ResQ Biometrics/ResQ-Biometrics-CNN/Models/model_ferCh_denseNet_1.h5'
    model = load_model(model_path)


    ### viz
    V = VisualizeShap(model, label_list)
    V.explain_layer(to_explain = X, layer_nr = 16)
