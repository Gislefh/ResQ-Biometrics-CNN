import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
class CustomMetrics:

    def __init__(self, embedding_size):
        self.embedding_size = embedding_size

    def triplet_accuracy(self, y_true, y_pred):

        return tf.constant(1, dtype='float32') 
        zero = tf.constant(0, dtype='float32')
        one = tf.constant(1, dtype='float32') 

        true = tf.constant(True, dtype='bool')

        def fun(input_):
            def y_is_1():
                dist1 = tf.norm(e2 - e3, ord=2) 
                dist2 = tf.norm(e1 - e3, ord=2)
                dist3 = tf.norm(e1 - e2, ord=2)
                if tf.math.less(dist1, dist2) is true:
                    if tf.math.less(dist1, dist3) is true:
                        return one
                else:
                    return zero 

            def y_is_2():
                dist1 = tf.norm(e1 - e3, ord=2) 
                dist2 = tf.norm(e1 - e2, ord=2)
                dist3 = tf.norm(e2 - e3, ord=2)
                if tf.math.less(dist1, dist2) is true:
                    if tf.math.less(dist1, dist3) is true:
                        return one
                else:
                    return zero 
            def y_is_3():
                dist1 = tf.norm(e1 - e2, ord=2) 
                dist2 = tf.norm(e1 - e3, ord=2)
                dist3 = tf.norm(e2 - e3, ord=2)
                if tf.math.less(dist1, dist2) is true:
                    if tf.math.less(dist1, dist3) is true:
                        return one
                else:
                    return zero 
            def ret_none():
                return tf.constant(17, dtype='float32')

            y_true_, y_pred_ = input_ 

            e1 = y_pred_[0:self.embedding_size]
            e2 = y_pred_[self.embedding_size:2*self.embedding_size]
            e3 = y_pred_[2*self.embedding_size:]
      
            y_true_ = tf.squeeze(y_true_)
            y_true_ = tf.dtypes.cast(y_true_, 'int32')
            acc = tf.switch_case(y_true_, branch_fns=[ret_none, y_is_1, y_is_2, y_is_3], default=ret_none)
            return acc
        
        acc = tf.map_fn(fun, (y_true, y_pred), dtype='float32')

        #return K.mean(acc)
        #return acc 
        return one