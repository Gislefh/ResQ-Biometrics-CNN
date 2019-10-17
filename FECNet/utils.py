from scipy.spatial import distance

def distances(prediction, embedding_size):
    e1 = prediction[:embedding_size]
    e2 = prediction[embedding_size:2*embedding_size]
    e3 = prediction[2*embedding_size:]

    d1 = distance.euclidean(e1, e2)
    d2 = distance.euclidean(e1, e3)
    d3 = distance.euclidean(e2, e3)
    
    print(d1,d2,d3)