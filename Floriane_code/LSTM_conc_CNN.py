import tensorflow.keras
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout, Flatten
from tensorflow.keras.layers import Input, ReLU, Dense, LSTM, concatenate, Activation, GRU, SimpleRNN
from tensorflow.keras.models import Model
#from keras.utils.constants import MAX_SEQUENCE_LENGTH_LIST, NB_CLASSES_LIST
#from utils.generic_utils import load_dataset_at
#from utils.keras_utils import train_model, evaluate_model, loss_model
#from layer_utils import AttentionLSTM
import numpy as np
import tensorflow.keras.losses


def LSTM_conc_CNN(X_train,nb_class):
    dim=np.shape(X_train)[1]
    print(dim)

    ip = Input(shape=(dim,1))
    NB_CLASS=nb_class
    #x = LSTM(128,return_sequences=True)(ip)
    #x = LSTM(64,return_sequences=True)(x)
    x = LSTM(128)(ip)
    x = Dropout(0.5)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 5, strides=1,padding='same')(y)#, kernel_initializer='GlorotUniform')(y)
    #y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(256, 3, strides=1, padding='same')(y)#, kernel_initializer='GlorotUniform')(y)
    #y = BatchNormalization()(y)
    y = Activation('relu')(y)

    #y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    #y = BatchNormalization()(y)
    #y = Activation('relu')(y)
    #y = Dropout(0.4)(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(NB_CLASS, activation='softmax')(x)
    #out=Dropout(0.2)(out)

    model = Model(ip, out)

    model.summary()

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.SGD(learning_rate=1e-4),
              metrics=['acc'])

    return model



	##
