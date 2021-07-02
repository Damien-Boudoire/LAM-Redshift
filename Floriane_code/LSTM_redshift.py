import numpy as np
import matplotlib.pyplot as plt
import collections
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Dropout, Conv1D, MaxPooling1D,Flatten, LSTM
from keras.models import Model, Sequential
from keras import metrics
import keras
import matplotlib.pyplot as plt




def LSTM_redshift(X_train,nb_class):
	#X_train=np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
	#X_test=np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

	if nb_class == 2 :
		loss='binary_crossentropy'
	else:
		loss='categorical_crossentropy'

	print(loss)
	
	dim=(17908,1)
	
	model_LSTM = Sequential()
	#model_LSTM.add(LSTM(500, return_sequences=True))
	model_LSTM.add(LSTM(128, return_sequences=True, input_shape=dim))
	#model_LSTM.add(LSTM(64, return_sequences=True))
	model_LSTM.add(LSTM(64))
	#model_LSTM.add(Dropout(0.1))
	model_LSTM.add(Dense(10, activation="tanh"))
	model_LSTM.add(Dense(nb_class, activation="sigmoid")) 
	model_LSTM.layers[0].trainable = False

	#print(model.summary())
	model_LSTM.compile(loss=loss,
              optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              #run_eagerly=True,
              metrics=['acc'])

	return model_LSTM


