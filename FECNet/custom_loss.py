import numpy as np
from scipy.spatial import distance
import tensorflow as tf
from tensorflow.python.keras import backend as K

class TripletLoss:
    
    def __init__(self, delta, embedding_size):
        self.delta = delta
        self.embedding_size = embedding_size

    def trip_loss(self, y_true, y_pred):
        zero = tf.constant(0, dtype = 'float32')

        def fun(input_):
            def y_is_1():
                dist1 = tf.math.square(tf.norm(e2 - e3, ord=2)) - tf.math.square(tf.norm(e1 - e3, ord=2)) + self.delta
                dist2 = tf.math.square(tf.norm(e2 - e3, ord=2)) - tf.math.square(tf.norm(e1 - e2, ord=2)) + self.delta 
                loss = tf.math.maximum(zero, dist1) + tf.math.maximum(zero, dist2)
                return loss
            def y_is_2():
                dist1 = tf.math.square(tf.norm(e1 - e3, ord=2)) - tf.math.square(tf.norm(e2 - e3, ord=2)) + self.delta
                dist2 = tf.math.square(tf.norm(e1 - e3, ord=2)) - tf.math.square(tf.norm(e1 - e3, ord=2)) + self.delta 
                loss = tf.math.maximum(zero, dist1) + tf.math.maximum(zero, dist2)
                return loss
            def y_is_3():
                dist1 = tf.math.square(tf.norm(e1 - e2, ord=2)) - tf.math.square(tf.norm(e1 - e3, ord=2)) + self.delta
                dist2 = tf.math.square(tf.norm(e1 - e2, ord=2)) - tf.math.square(tf.norm(e2 - e3, ord=2)) + self.delta 
                loss = tf.math.maximum(zero, dist1) + tf.math.maximum(zero, dist2)
                return loss

            def ret_none():
                return tf.constant(17, dtype='float32')

            y_true_, y_pred_ = input_ 

            e1 = y_pred_[0:self.embedding_size]
            e2 = y_pred_[self.embedding_size:2*self.embedding_size]
            e3 = y_pred_[2*self.embedding_size:]
      
            y_true_ = tf.squeeze(y_true_)
            y_true_ = tf.dtypes.cast(y_true_, 'int32')
            loss = tf.switch_case(y_true_, branch_fns=[ret_none, y_is_1, y_is_2, y_is_3], default=ret_none)
            return loss

        loss = tf.map_fn(fun, (y_true, y_pred), back_prop=True, infer_shape=False, dtype='float32')

        return K.mean(loss) #scalar or embedding size? - now using scalar

      

if __name__ == '__main__':
    import tensorflow as tf
    import numpy as np
    def Custom_loss(y,outputs):
        hold_loss = []
        for exp,pred in zip(y,outputs):
            if exp >= pred:
                result = tf.pow(pred * 0.5,2) - exp
                hold_loss.append(result)
            else:
                hold_loss.append(tf.subtract(pred-exp))
        return tf.reduce_mean(hold_loss)

    L = TripletLoss(delta=0.1, embedding_size=24)
    
    np_x = np.random.randn(16,24*3)
    np_x = np_x.astype(np.float32)
    np_y = np.random.randint(3,4, size= (16, 1))

    x = tf.constant(np_x)
    y = tf.constant(np_y)

    with tf.Session() as sess:
        print(sess.run(L.trip_loss(y, x)))