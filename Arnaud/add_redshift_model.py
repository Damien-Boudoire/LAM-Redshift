# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 15:59:21 2021

@author: Arnaud
"""

from keras.models import Model
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Concatenate, Input


def new_model(input_shape, num_classes):
    
    # Definition des 2 entrees
    Pdf = Input(shape=input_shape)
    Redshift = Input(shape=(1,))
    
    # CNN pour les Pdfs
    x = Conv1D(filters = 64, kernel_size = 12, padding='same')(Pdf)
    x = MaxPooling1D(pool_size=2, strides=2,  padding="same")(x)
    
    x = Conv1D(filters = 128, kernel_size = 6, activation = 'relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2,  padding="same")(x)
    
    x = Conv1D(filters = 256, kernel_size = 3, activation = 'relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2,  padding="same")(x)
    
    x = Flatten()(x)
    
    CNN_out = Dense(128, activation='relu')(x)


    # On concatene la sortie du CNN avec le Redshiftd'entree
    merged = Concatenate()([CNN_out, Redshift])
    
    merged = Dense(64, activation='relu')(merged)
    
    output = Dense(num_classes, activation='softmax')(merged)
    
    # On definitit le model avec 2 entrées
    model = Model(inputs=[Pdf, Redshift], outputs=output)
    
    return model
    
    

    