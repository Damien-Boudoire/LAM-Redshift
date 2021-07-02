import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import collections
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, MaxPooling1D,Flatten, LSTM, TimeDistributed, Embedding, GRU, Activation, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import metrics
import tensorflow.keras
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam, Adamax,RMSprop, Adagrad

from tensorflow.keras import activations



def LSTM_CNN_redshift(learning_rate=1e-4):
    dim=(17908, 1)#,np.shape(X_train)[1:]
    nb_class=2
    print(learning_rate)
    #print('dropout_rate : ' + str(dropout_rate))



    if nb_class == 2:
      loss='binary_crossentropy'
    else :
      loss='categorical_crossentropy'

    print(loss)

    model= Sequential()
    model.add(Conv1D(32,kernel_size=10,padding='same', activation='relu', input_shape=dim))
    model.add(Conv1D(32,kernel_size=10,padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2)) 

    model.add(Conv1D(64,kernel_size=6,padding='same', activation='relu'))
    model.add(Conv1D(64,kernel_size=6,padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))   

    model.add(Conv1D(128,kernel_size=5,padding='same',activation='relu'))
    model.add(Conv1D(128,kernel_size=5,padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(256,kernel_size=3,padding='same',activation='relu'))
    model.add(Conv1D(256,kernel_size=3,padding='same',activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(32))
    #model.add(Flatten())
    model.add(Dense(nb_class,activation='softmax'))

    #model.summary()
    model.compile(loss=loss,
              optimizer=tensorflow.keras.optimizers.Adam(learning_rate=learning_rate),
              metrics=['acc'])

    return model


#LSTM_CNN_redshift()
	##
