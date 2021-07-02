# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 11:43:40 2021

@author: Arnaud
"""

import tensorflow as tf
from tensorflow.python.keras.backend import set_session

def config_gpu_memory():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    set_session(tf.compat.v1.Session(config=config))