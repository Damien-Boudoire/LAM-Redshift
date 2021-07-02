from dataset_redshift import dataset_redshift
from dataset_commun_redshift import dataset_commun_redshift
from MLP_redshift import MLP_redshift
from CNN1D_redshift import CNN1D_redshift
from LSTM_redshift import LSTM_redshift
from LSTM_CNN_redshift import LSTM_CNN_redshift
from LSTM_conc_CNN import LSTM_conc_CNN
from report import *
from pdfReport import *
from heatmap import *
#import model_inception
#from visu_data import visu_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import collections
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from imblearn.under_sampling import RandomUnderSampler
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from tensorflow.keras.initializers import RandomNormal, TruncatedNormal, Zeros, Ones, GlorotNormal, GlorotUniform, Identity, Orthogonal

#tf.config.list_physical_devices(device_type='GPU')

from config_gpu_memory import config_gpu_memory

config_gpu_memory()

print('hello')
nom_class='DeltaZ'
Undersample=True
X_train, X_validation ,X_test, target_train, target_validation ,target_test , Y_train , Y_validation, Y_test , nb_class, nom_classes = dataset_redshift(nom_class,Undersample=Undersample)

#X_train, X_validation ,X_test, target_train, target_validation ,target_test , Y_train , Y_validation, Y_test , nb_class, nom_classes = dataset_commun_redshift(nom_class,Undersample=Undersample)


X_test=np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
X_train=np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))	
X_validation=np.reshape(X_validation, (X_validation.shape[0],X_validation.shape[1],1))	


model=LSTM_CNN_redshift()


#reduce_lr = ReduceLROnPlateau(factor=0.5, cooldown=1,
#                              patience=5, min_lr=0.000000001)


def scheduler(epoch, lr):
     if epoch < 10:
         return lr
     else:
         return lr * tf.math.exp(-0.1)

reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

epoch=80
batch_size=128


model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint("best_model.h5",
    monitor='val_acc',
    save_best_only=True)


h = model.fit(X_train, Y_train,
               epochs=epoch,
               batch_size=batch_size,
               verbose =1,
               callbacks=[reduce_lr, model_checkpoint_callback],
               validation_data = (X_validation , Y_validation))



h=model.load_model('best_model.h5')

print(h.history['lr'])

y_pred_test=model.predict(X_test)
y_pred_test=np.argmax(y_pred_test, axis=1)

conf_matrix_test=confusion_matrix(target_test, y_pred_test)

y_pred_train=model.predict(X_train)
y_pred_train=np.argmax(y_pred_train, axis=1)


conf_matrix_train=confusion_matrix(target_train, y_pred_train)

report=classification_report(target_test, y_pred_test,output_dict=True)


acc=h.history['acc']
loss=h.history['loss']
val_acc=h.history['val_acc']
val_loss=h.history['val_loss']

if Undersample:
    Undersample='Undersample'
else:
    Undersample='Unballanced'

stringlist = []
model.summary(print_fn=lambda x: stringlist.append(x))
short_model_summary = "\n".join(stringlist)

dicte={}
dicte["conf_matrix_train"] = conf_matrix_train
dicte["conf_matrix_test"] = conf_matrix_test
dicte["nb_class"]=nb_class
dicte["nom_classes"]=nom_classes
dicte["acc"]=acc
dicte["loss"]=loss
dicte["val_acc"]=val_acc
dicte["val_loss"]=val_loss
dicte["report"]=report
dicte["model_summary"]=short_model_summary

nom = nom_class + '_' + Undersample + '_epoch' + str(epoch)+ '_BS' + str(batch_size) + '_reduceLRexpo_otherdata_modelcheckpoint'
np.save(nom, dicte)