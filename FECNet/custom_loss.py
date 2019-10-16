import numpy as np
from scipy.spatial import distance
import tensorflow as tf
from tensorflow.python.keras import backend as K

class TripletLoss:
    
    def __init__(self, delta, embedding_size):
        self.delta = delta
        self.embedding_size = embedding_size




    """
    -- Triplet loss function -- 
    gt = 1 or 2 or 3:
        for gt = 1 => e1 and e2 is the most similar
        for gt = 2 => e2 and e3 is the moset similar    ### assuming this is correct - check that
        for gt = 3 => e1 and e3 is the moset similar    ### assuming this is correct - check that

    e_{1,2,3} - embedding

    delta - a small margin
    """
    def trip_loss(self, y, y_pred):
        # split the output from the model
        e1 = y_pred[:, 0:self.embedding_size]
        e2 = y_pred[:, self.embedding_size:2*self.embedding_size]
        e3 = y_pred[:, 2*self.embedding_size:]


        y = 1
        if y == 1: ## TODO not sure if the if's are correct- don't think so
            dist1 = tf.math.square(tf.norm(e1 - e2, ord=2)) - tf.math.square(tf.norm(e1 - e3, ord=2)) + self.delta
            dist2 = tf.math.square(tf.norm(e1 - e2, ord=2)) - tf.math.square(tf.norm(e2 - e3, ord=2)) + self.delta 
            loss = tf.math.maximum(0.0, dist1) + tf.math.maximum(0.0, dist2)

        elif y == 2:
            dist1 = distance.euclidean(e2, e3)**2 - distance.euclidean(e1, e3)**2 + self.delta
            dist2 = distance.euclidean(e2, e3)**2 - distance.euclidean(e1, e2)**2 + self.delta
            loss = np.maximum(0, dist1) + np.maximum(0, dist2)
        
        elif y == 3:
            dist1 = distance.euclidean(e1, e3)**2 - distance.euclidean(e2, e3)**2 + self.delta
            dist2 = distance.euclidean(e1, e3)**2 - distance.euclidean(e1, e2)**2 + self.delta
            loss = np.maximum(0, dist1) + np.maximum(0, dist2)

        else:
            print('--- Ground Truth not in the scope ---')
            return None
        
        return K.mean(loss)

    """ FROM GIT - faceNet

    def triplet_loss(y_true, y_pred):
        a_pred = y_pred[:, 0:128]
        p_pred = y_pred[:, 128:256]
        n_pred = y_pred[:, 256:384]
        positive_distance = K.square(tf.norm(a_pred - p_pred, axis=-1))
        negative_distance = K.square(tf.norm(a_pred - n_pred, axis=-1))
        loss = K.mean(K.maximum(0.0, positive_distance - negative_distance + alpha))
        return loss
    """


