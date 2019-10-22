from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def distances(prediction, embedding_size):
    e1 = prediction[:embedding_size]
    e2 = prediction[embedding_size:2*embedding_size]
    e3 = prediction[2*embedding_size:]

    d1 = distance.euclidean(e1, e2)
    d2 = distance.euclidean(e1, e3)
    d3 = distance.euclidean(e2, e3)
    
    print(d1,d2,d3)


""" PCA for MNIST digits dataset
calculates pa
"""
def pca(predictions, embedding_size, y, N_comp = 2):
    from sklearn.decomposition import PCA
    predictions = np.squeeze(np.array(predictions))

    output_vec = np.zeros((len(predictions)*3, embedding_size) )

    for i in range(len(predictions)):
        output_vec[0 + i*3, :]  = predictions[i, :embedding_size]
        output_vec[1 + i*3, :] = predictions[i, embedding_size:2*embedding_size]
        output_vec[2 + i*3, :] = predictions[i, 2*embedding_size:]


    pca = PCA(n_components = N_comp)
    reduced_pred = pca.fit_transform(output_vec)
    


    y = y[:len(output_vec)]
    #label_list = ['rx', 'bx', 'kx', 'mx', 'yx', 'ro', 'bo', 'ko', 'mo', 'yo'] # 0,1,...,9

    #label_list = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255 , 0], [0, 255, 255], [120, 0, 120], [0, 120, 120], [120, 120, 0], [120, 120, 120]]
    labels = []
    for i in range(len(y)):
        cmap = plt.cm.Spectral(y[i]/10)
        labels.append(cmap)
    #labels = np.array(labels)


    if N_comp == 2: ## 2d plot
        plt.plot(reduced_pred[:, 0], reduced_pred[:, 1], labels)
        plt.show()

    if N_comp == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xs =output_vec[:, 0], ys = output_vec[:, 1], zs = output_vec[:, 2], zdir='z', s=20, c=labels)
        plt.show()



