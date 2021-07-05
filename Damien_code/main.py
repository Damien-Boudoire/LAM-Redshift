import keras.models

from model.CnnGruModel import makeCnnGRUModel, makeCnnGruModelAlt, makeCnnGruBig
from dataset_commun_redshift import dataset_commun_redshift
from report.report import testAndReport
from report.Save_in_dict import Save_in_dict

import numpy as np
import sys
import tensorflow

from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

from numpy.random import seed

from config_gpu_memory import config_gpu_memory

seed(1)
tensorflow.random.set_seed(1)

config_gpu_memory()

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Bad parameters need 4 : epochs batch target balanced")
        exit(-1)

    try:
        epochs = int(sys.argv[1])
        batch = int(sys.argv[2])
        target = sys.argv[3]
        balanced = True if int(sys.argv[4]) == 1 else False

        X_train, X_validation, X_test,\
        target_train, target_validation,\
        target_test, Y_train, Y_validation,\
        Y_test, nb_classes, nom_classes = dataset_commun_redshift(target, balanced)

        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_validation = np.reshape(X_validation, (X_validation.shape[0], X_validation.shape[1], 1))

        categorical = nb_classes > 2
        loss_fn = "categorical_crossentropy" if categorical else "BinaryCrossentropy"

        modelFile = "GRU/saved/models/{0}_{1}_best_model.h5".format(target, balanced)

        lr = 1e-4
        decay = 0.75
        step = 5 #10
        lr_scheduler = LearningRateScheduler(lambda epoch: lr * (decay ** np.floor(epoch / step)))
                      #ReduceLROnPlateau(factor=.9, patience=3, verbose=1, min_lr=1e-10)
        checkpoint = ModelCheckpoint(modelFile, save_best_only=True, monitor="val_loss")
        stopping = EarlyStopping(monitor="val_loss", patience=10, verbose=1)

        units = 256
        depth = 3
        if target == "Success3":
            depth=5
        dropout = 0

        rowShape = X_train[0].shape

    #    gru = makeCnnGruModelAlt(input=dataset.getRowShape(), output=dataset.nbClasses, dropout=dropout)
        gru = makeCnnGruBig(units=units, input=rowShape, output=nb_classes, depth=1, reduce=2, dropout=dropout)
        gru.compile("Adam", loss_fn, metrics=["acc"])

        callbacks = [lr_scheduler, checkpoint] #, stopping

        y_train = Y_train if categorical else target_train
        y_validation = Y_validation if categorical else target_validation

        print("to fit")
        history = gru.fit(X_train, y_train,
                          validation_data=(X_validation, y_validation),
                          epochs=epochs, batch_size=batch, callbacks=callbacks)
        print("fitted")
        gru = keras.models.load_model(modelFile)
        nameModel = "BigCnnGRU_{0}_{1}_{2}_{3}_{4}".format(units, depth, epochs, target, balanced)
        Save_in_dict(gru, X_test, Y_test, X_train , Y_train,
                     history, nb_classes, nom_classes, "CnnGRU")
        testAndReport(nameModel, gru, nom_classes, history, X_train, Y_train, X_test, Y_test)

    except Exception as e:
        print("Bad parameters", e)
