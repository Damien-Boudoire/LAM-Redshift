# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 10:26:26 2021

@author: Arnaud
"""

from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten


def CNN_model1(input_shape, num_classes):
    
    model = Sequential()
    
    model.add(Conv1D(filters = 32, kernel_size = 9, activation = 'relu', input_shape = input_shape))
    model.add(MaxPooling1D(pool_size=2))  
    
    model.add(Conv1D(filters = 64, kernel_size = 6, activation = 'relu', input_shape = input_shape))
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Conv1D(filters = 128, kernel_size = 3, activation = 'relu'))
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Flatten())
    
    model.add(Dense(64, activation='relu'))
    
    model.add(Dense(32, activation='relu'))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    return model