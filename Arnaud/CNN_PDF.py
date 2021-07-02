# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 17:38:27 2021

@author: Arnaud
"""

import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, Callback
from keras.models import load_model
#import keras.backend as K

#from dataset_redshift import dataset_redshift
from dataset_commun_redshift import dataset_commun_redshift
#from dataset_redshift import dataset_redshift
from plot_learning_curves import plot_learning_curves
from plot_confusion_matrix import plot_CM
from config_gpu_memory import config_gpu_memory
from CNN_model1 import CNN_model1
#from RedshiftDataset import RedshiftDataset
#from report import testAndReport
from Save_in_dict import Save_in_dict


config_gpu_memory()

#for Data in ['DeltaZ', 'Success2', 'Success3', 'Flag3']:
for Data in ['DeltaZ']:    
    
    # Les cibles peuvent être : 'Flag' , 'Success_2' , 'Success_3', 'Flag_class', 'DeltaZ'
    # Les donnees d'entrees X peuvent etre : 'logPdf', 'Pdf' et 'logVraisemblance'
    X_train, X_valid, X_test, target_train, target_valid, target_test , Y_train, Y_valid, Y_test , nb_classes, nom_classes = dataset_commun_redshift(Data, Undersample=False)
    # dataset = RedshiftDataset("./data", False, target)
    X_train = np.exp(X_train)
    X_valid = np.exp(X_valid)
    X_test = np.exp(X_test)
    # reshape input to be [samples, time steps, features = 1]
    X_train_cnn = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_valid_cnn = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1], 1))
    X_test_cnn = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    model = CNN_model1(X_train_cnn.shape[1:], nb_classes)

    # compile the model - use categorical crossentropy, and the adam optimizer
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])

    epochs = 40
    batch_size = 128


    callbacks = [
        ModelCheckpoint(
            "best_model.h5", save_best_only=True, monitor="val_loss"
            ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=0.00001
            ),
        EarlyStopping(monitor="val_loss", patience=5, verbose=1),

    ]

    h = model.fit(X_train_cnn, Y_train,
              epochs=epochs,
              batch_size=batch_size,
              callbacks=callbacks,
              verbose =1,
              validation_data = (X_valid_cnn, Y_valid))





    model = load_model("best_model.h5")
    model.evaluate(X_test_cnn, Y_test, verbose =2)



    plot_CM(model, X_train_cnn, target_train, nom_classes, 'CM_train.png')
    plot_CM(model, X_test_cnn, target_test, nom_classes, 'CM_test.png')

#plot_learning_curves(h)

    Save_in_dict(model, X_test_cnn, target_test, X_train_cnn, target_train, h, nb_classes, nom_classes, 'CNN_PDF_' + Data)
# testAndReport('CNN', model, nom_classes, h, X_train_cnn, target_train, X_test_cnn, target_test)
