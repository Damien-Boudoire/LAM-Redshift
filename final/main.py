import sys

from numpy.random import seed
import tensorflow as tf
from tensorflow.python.keras.backend import set_session

from models import make_CNN, makeCnnGru, makeLSTMCNN

config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    set_session(tf.compat.v1.Session(config=config))

seed(1)
tensorflow.random.set_seed(1)

if __name__ == '__main__':
    try:
        modelName = sys.argv[1]
        target = sys.argv[2]
        preprocessedPath = sys.argv[3]
    except Exception as error:
        print(error)