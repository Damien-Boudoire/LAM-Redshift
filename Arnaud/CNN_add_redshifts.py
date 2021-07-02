# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 17:38:27 2021

@author: Arnaud
"""

import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.models import load_model


from dataset_redshift_add_redshifts import dataset_redshift
from plot_learning_curves import plot_learning_curves
from plot_confusion_matrix import plot_CM_Redshift
from config_gpu_memory import config_gpu_memory
from add_redshift_model import new_model

config_gpu_memory()

# Les cibles peuvent être : 'Flag' , 'Success_2' , 'Success_3', 'Flag_class', 'DeltaZ'
# Les donnees d'entrees X peuvent etre : 'logPdf', 'Pdf' et 'logVraisemblance'
X_train, X_test, target_train, target_test , Y_train , Y_test , Redshifts_train, Redshifts_test, nb_classes, nom_classes = dataset_redshift('Flag_class', 0.15)


# reshape input to be [samples, time steps, features = 1]
X_train_cnn = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test_cnn = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

model = new_model(X_train_cnn.shape[1:], nb_classes)

# compile the model - use categorical crossentropy, and the adam optimizer
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

epochs = 40
batch_size = 32

callbacks = [
    ModelCheckpoint(
        "best_model.h5", save_best_only=True, monitor="val_loss"
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=0.0001
    ),
    EarlyStopping(monitor="val_loss", patience=8, verbose=1),
]


h = model.fit([X_train_cnn, Redshifts_train], Y_train,
              epochs=epochs,
              batch_size=batch_size,
              callbacks=callbacks,
              verbose =1,
              validation_split=0.15)

model = load_model("best_model.h5")
model.evaluate([X_test_cnn, Redshifts_test], Y_test, verbose =2)

# plot_CM_Redshift(model, X_train_cnn, Redshifts_train, target_train, nom_classes, 'CM_train.png')
plot_CM_Redshift(model, X_test_cnn, Redshifts_test, target_test, nom_classes, 'CM_test.png')

plot_learning_curves(h)

