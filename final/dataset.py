import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

data_path = "./data/"

def load_processed(target, undersample):
    if undersample:
        eq = 'balanced'
    else:
        eq = 'unbalanced'

    if target == 'DeltaZ':
        nb_class = 2
        nom_classes = ["Deltaz<1e-3", "Deltaz>=1e-3"]

    if target == 'Success2':
        nb_class = 2
        nom_classes = ["Success", "Spurious/Mismatch"]

    if target == 'Flag3':
        nb_class = 3
        nom_classes = ["Flag 1", "Flags 2-9", "Flags 3-4"]

    if target == 'Success3':
        nb_class = 3
        nom_classes = ["Success", "Spurious", "Mismatch"]

    name = data_path + target + '/' + eq
    print(name)
    X_train = np.load(name + '/TRAIN/X.npy')
    target_train = np.load(name + '/TRAIN/Y.npy')

    X_test = np.load(name + '/TEST/X.npy')
    target_test = np.load(name + '/TEST/Y.npy')

    X_validation = np.load(name + '/VALID/X.npy')
    target_validation = np.load(name + '/VALID/Y.npy')

    Y_train = tf.keras.utils.to_categorical(target_train, nb_class)  # convertir en one-hot-code
    Y_test = tf.keras.utils.to_categorical(target_test, nb_class)
    Y_validation = tf.keras.utils.to_categorical(target_validation, nb_class)

    return X_train, X_validation, X_test, target_train, target_validation, target_test, Y_train, Y_validation, Y_test, nb_class, nom_classes


def load_dataset(target,undersample):

  ##### Target #######
  # Les cibles peuvent être : 'Flag' , 'Success' , 'DeltaZ'

	pdf_zgrid=np.load(data_path + 'zgrid32.npy')
	attributes=np.load(data_path + 'attributes32.npy', allow_pickle=True)
	pdfs=np.load(data_path + 'pdfs32.npy', allow_pickle=True)


	if target == 'Flag':
		y=np.floor(attributes[:,-1]).astype(int)
		y=np.where(y==9, 0, y)

		nom_classes = ["Flag 9", "Flag 1", "Flag 2", "Flag 3", "Flag 4"]


	elif target == 'Success2':
		flag=attributes[:,5]
		y=[]

		for i in flag:
			if i =='success':
				nw_fl=0
			else:
				nw_fl=1

			y.append(nw_fl)

		nom_classes = ["Success", "Spurious/Mismatch"]




	elif target == 'Flag3':
		flag=np.floor(attributes[:,-1]).astype(int)
		y=[]

		for i in flag:

			if (i == 2) or (i == 9):
				nw_fl=1
			elif (i == 3) or (i == 4):
				nw_fl=2
			elif i == 1:
				nw_fl=0

			y.append(nw_fl)

		nom_classes = ["Flag 1", "Flags 2-9", "Flags 3-4"]


	elif target == 'DeltaZ':
		flag=attributes[:,3]

		y=[]

		flag=abs(flag)

		for i in flag:

			if i < 5e-3:
				nw_fl=0
			elif i >= 5e-3:
				nw_fl=1

			y.append(nw_fl)

		nom_classes= ["Deltaz<1e-3", "Deltaz>=1e-3"]

	elif target == 'Success3':
		flag=attributes[:,5]
		y=[]

		for i in flag:
			if i =='success':
				nw_fl=0
			elif i == 'spurious':
				nw_fl=1
			else:
				nw_fl=2

			y.append(nw_fl)

		nom_classes = ["Success", "Spurious" , "Mismatch"]

	else:
		raise ValueError("Choisis la target = 'Flag' , 'Success2' , 'Flag3' , 'DeltaZ' , 'Success3' ")


	nb_classes=len(np.unique(y))

	X_train, X_test, target_train, target_test = train_test_split(pdfs, y, test_size=0.15)
	X_train, X_validation, target_train, target_validation = train_test_split(X_train, target_train, test_size=0.15)

	if undersample :
		rus = RandomUnderSampler(random_state=0)
		X_train, target_train = rus.fit_resample(X_train, target_train)
		X_test, target_test = rus.fit_resample(X_test, target_test)
		X_validation, target_validation = rus.fit_resample(X_validation, target_validation)


	Y_train =  tf.keras.utils.to_categorical(target_train, nb_classes)   #convertir en one-hot-code
	Y_test = tf.keras.utils.to_categorical(target_test, nb_classes)
	Y_validation = tf.keras.utils.to_categorical(target_validation, nb_classes)



	return X_train, X_validation ,X_test, target_train, target_validation ,target_test , Y_train , Y_validation, Y_test , nb_classes, nom_classes