from imblearn.under_sampling import RandomUnderSampler
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import collections
from sklearn.model_selection import train_test_split
#from tensorflow.keras import np_utils

def dataset_commun_redshift(target,Undersample=False):

	if Undersample:
		eq='balanced'
	else:
		eq='unbalanced'


	if target=='DeltaZ':

		nb_class=2
		nom_classes= [r"$\Delta$Z $< 10^{-3}$", r"$\Delta$Z $\geq 10^{-3}$"]
		eq='unbalanced'
	
	if target=='Success2':

		nb_class=2
		nom_classes= ["Success", "Spurious/Mismatch"]

	if target=='Flag3':

		nb_class=3
		nom_classes = ["flags 1", "flags 2-9", "flags 3-4"]

	if target=='Success3':

		nb_class=3
		nom_classes = ["Success", "Spurious" , "Missmatch"]



	name = '../data/' + target + '/' + eq
	print(name)
	X_train=np.load(name + '/TRAIN/X.npy')
	target_train=np.load(name + '/TRAIN/Y.npy')

	X_test=np.load(name + '/TEST/X.npy')
	target_test=np.load(name + '/TEST/Y.npy')

	X_validation=np.load(name + '/VALID/X.npy')
	target_validation=np.load(name + '/VALID/Y.npy')

	Y_train =  tf.keras.utils.to_categorical(target_train, nb_class)   #convertir en one-hot-code
	Y_test = tf.keras.utils.to_categorical(target_test, nb_class)
	Y_validation = tf.keras.utils.to_categorical(target_validation, nb_class)

	return X_train, X_validation ,X_test, target_train, target_validation ,target_test , Y_train , Y_validation, Y_test , nb_class, nom_classes