import tensorflow as tf
import os


## not working yet

class CustomCallback(tf.compat.v2.keras.callbacks.TensorBoard):

    def __init__(self, log_dir):
        self.log_dir = log_dir
        
        self.__get_writer('writer_1') 
    
    def on_epoch_end(self, epoch, logs=None):

        name = 'test___loss'
        value = logs.get('loss')
        with writer.as_default():
            tf.python.ops.summary_ops_v2.scalar(name, value, step=epoch)


    def __get_writer(self, writer_name):

        path = os.path.join(self.log_dir, writer_name)
        writer = tf.python.ops.summary_ops_v2.create_file_writer_v2(path)
        self._writers[writer_name] = writer
        return self._writers[writer_name]

