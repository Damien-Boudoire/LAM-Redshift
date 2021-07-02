import numpy as np
import matplotlib.pyplot as plt
import collections
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Dropout, Conv1D, MaxPooling1D,Flatten
from keras.models import Model, Sequential
from keras import metrics
import matplotlib.pyplot as plt




def CNN1D_redshift(X_train, nb_class):

	if nb_class == 2 :
		loss='binary_crossentropy'
	else:
		loss='categorical_crossentropy'

	dim=np.shape(X_train)[1:]
	print(dim, loss)
	
	model = Sequential()
	model.add(Conv1D(filters=64, kernel_size=(5), strides=1,activation='tanh', input_shape=dim))
	model.add(Conv1D(filters=128, kernel_size=(5), strides=1,activation='tanh'))
	#model.add(Conv1D(filters=256, kernel_size=(15), strides=3,activation='tanh'))
	#model.add(MaxPooling1D(pool_size=2, strides=1, padding="same"))
	#model.add(Conv1D(filters=500, kernel_size=(15), strides=3,activation='tanh'))
	#model.add(MaxPooling1D(pool_size=2, strides=1, padding="same"))
	model.add(Dropout(0.1))
	model.add(Flatten())
	#model.add(Dense(128, activation='relu'))
	model.add(Dense(10, activation='relu'))
	model.add(Dense(nb_class, activation='sigmoid'))

	#print(model.summary())
	model.compile(loss=loss,
              optimizer='adam',
              metrics=['acc'])

	return model