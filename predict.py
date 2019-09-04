import sys
sys.path.insert(0, "classes")
import keras
from keras.models import load_model
from class_generator import Generator
from class_predict import Predict


### consts
test_path = 'C:\\Users\\47450\\Documents\\ResQ Biometrics\\Data sets\\face-expression-recognition-dataset\\images\\validation'
N_channels = 1
batch_size = 16
image_shape = (60, 60)
N_classes = 6
X_shape = (batch_size, image_shape[0], image_shape[1], N_channels)
Y_shape = (batch_size, N_classes)

## create ganerator
test_gen = Generator(test_path, X_shape, Y_shape, N_classes, N_channels, batch_size)
test_gen = test_gen.generator_from_dir(include_folder_list = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise'], N_images_per_class = 700)




model = load_model("Models\\test_model.h5")
prd = Predict(model, labels = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise'])
#prd.pred_from_cam()

prd.conf_matrix(test_gen, 700*6)