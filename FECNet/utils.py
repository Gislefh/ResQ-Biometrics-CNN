from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

## returns distanses
def Distances(prediction, embedding_size):
    e1 = prediction[:embedding_size]
    e2 = prediction[embedding_size:2*embedding_size]
    e3 = prediction[2*embedding_size:]

    d1 = distance.euclidean(e1, e2)
    d2 = distance.euclidean(e1, e3)
    d3 = distance.euclidean(e2, e3)
    
    #print(d1,d2,d3)
    return [d1, d2, d3]

""" PCA for MNIST digits dataset
calculates pa
"""
def pca_MNIST(predictions, embedding_size, y, N_comp = 2):
    predictions = np.squeeze(np.array(predictions))

    output_vec = np.zeros((len(predictions)*3, embedding_size) )

    for i in range(len(predictions)):
        output_vec[0 + i*3, :]  = predictions[i, :embedding_size]
        output_vec[1 + i*3, :] = predictions[i, embedding_size:2*embedding_size]
        output_vec[2 + i*3, :] = predictions[i, 2*embedding_size:]


    pca = PCA(n_components = N_comp)
    reduced_pred = pca.fit_transform(output_vec)
    


    y = y[:len(output_vec)]

    labels = []
    for i in range(len(y)):
        cmap = plt.cm.Spectral(y[i]/10)
        labels.append(cmap)

    if N_comp == 2: ## 2d plot
        plt.scatter(reduced_pred[:, 0], reduced_pred[:, 1], c=labels)
        plt.show()

    if N_comp == 3: ## 3d plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xs =output_vec[:, 0], ys = output_vec[:, 1], zs = output_vec[:, 2], zdir='z', s=20, c=labels)
        plt.show()


""" PCA for FECNet data
images: list containig the images used in the prediction
"""
def pca_FECNet(predictions, embedding_size, images, N_comp = 2):
    predictions = np.squeeze(np.array(predictions))

    output_vec = np.zeros((len(predictions)*3, embedding_size) )

    for i in range(len(predictions)):
        output_vec[0 + i*3, :]  = predictions[i, :embedding_size]
        output_vec[1 + i*3, :] = predictions[i, embedding_size:2*embedding_size]
        output_vec[2 + i*3, :] = predictions[i, 2*embedding_size:]

    size = 0.01

    pca = PCA(n_components = N_comp)
    reduced_pred = pca.fit_transform(output_vec)
    #plt.plot(reduced_pred[:,0], reduced_pred[:, 1], 'rx')
    #plt.show()
    #return None

    fig, ax = plt.subplots(1, 1)
    ax.set_xlim(np.amin(reduced_pred[:,0])-size, np.amax(reduced_pred[:,0])+size)
    ax.set_ylim(np.amin(reduced_pred[:,1])-size, np.amax(reduced_pred[:,1])+size)
    #print(np.amax(reduced_pred), np.amin(reduced_pred))
    
    for i, im in enumerate(images):
        extent = [reduced_pred[i, 0]-size, reduced_pred[i, 0]+size, reduced_pred[i, 1]-size, reduced_pred[i, 1]+size]
        #print(extent)
        ax.imshow(im, extent=extent)
    plt.show()


## output from generator sould be batch_size = 1
## uses distance from above
def eval_gen(gen, model, data_len, embedding_size):
    acum = 0
    cnt = 0
    for i, (x,y) in enumerate(gen):
        cnt +=1
        print('predicted on ', cnt, '/ ',data_len, ' samples')
        pred = model.predict(x, batch_size = 1, steps = 1)
        d1, d2, d3 = distances(np.squeeze(pred), embedding_size)

        if d1 < d2 and d1 < d3:
            gt = 3
        elif d2 < d1 and d2 < d3:
            gt = 2
        elif d3 < d1 and d3 < d2:
            gt = 1
        else:
            print('what?!')
            exit()

        if gt == y:
            acum += 1
        if cnt > data_len:
            break
    print(acum/cnt)
        
