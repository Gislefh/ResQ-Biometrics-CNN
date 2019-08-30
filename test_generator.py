import sys
sys.path.insert(0, "classes")
from class_generator import Generator




path = 'C:\\Users\\47450\\Documents\\ResQ Biometrics\\Data sets\\FEC_dataset\\faceexp-comparison-data-train-public.csv'
N_channels = 3
batch_size = 32
image_shape = (120, 100)
N_classes = 2
X_shape = (batch_size, image_shape[0], image_shape[1], N_channels)
Y_shape = (batch_size, N_classes)


gen = Generator(path, X_shape, Y_shape, N_classes, N_channels, batch_size)
gen.generator_from_web()