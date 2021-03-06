import sys
import keras
import numpy as np

from numpy.random import seed
import tensorflow as tf
from tensorflow.python.keras.backend import set_session

from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

import dataset
from models import make_CNN_3layers, make_CNN_8layers, make_CNNGRU, make_CNNLSTM
from report.save_in_dict import save_in_dict

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.compat.v1.Session(config=config))

seed(1)
tf.random.set_seed(1)

#dimension of our data
input_shape = (17908, 1)

### optimizer parameters
optimizer = "adam"
loss_function = "categorical_crossentropy"
metrics = ["acc"]
###

### list of tested learning rate update functions
learning_rate_schedulers = [
    LearningRateScheduler(lambda epoch: 1e-4 * tf.math.exp(-.1 * epoch)),
    LearningRateScheduler(lambda epoch, lr: lr if epoch < 10 else lr * tf.math.exp(-0.1)),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=0.000001)]
###


if __name__ == '__main__':
    ### main function of this application
    #   usage : python main.py model_name target_name (raw | preprocessed) (full | undersampled) epoch batch
    #
    #   model_name = CNN3 | CNN8 | CNNGRU | CNNLSTM
    #       the model to train and test
    #   target_name : DeltaZ | Success2 | Success3 | Flag3
    #        labels to use for classification
    #   raw : build the dataset from the original files
    #   preprocessed : use the built datasets in data/
    #   undersampled : remove data to get balanced classes
    #   full : do not remove data
    #   epoch : number of epoch that the training will last
    #   batch : size of batch used
    ###

    try:
        modelName = sys.argv[1]
        if modelName not in ["CNN3", "CNN8", "CNNGRU", "CNNLSTM"]:
            print(modelName)
            raise Exception
        target = sys.argv[2]
        if target not in ["DeltaZ", "Success2", "Success3", "Flag3"]:
            print(target)
            raise Exception
        preprocessed = sys.argv[3]
        if preprocessed not in ["preprocessed", "raw"]:
            print(preprocessed)
            raise Exception
        undersampled = sys.argv[4]
        if undersampled not in ["undersampled", "full"]:
            print(undersampled)
            raise Exception
        epoch = int(sys.argv[5])
        batch_size = int(sys.argv[6])
    except Exception:
        print("Bad arguments \n got : {0} \n need : model target preprocessed undersampled epoch batch".format(sys.argv))
        exit(-1)


    # we load the data using different function depending we are using preprocessed dataset or not
    if preprocessed == "preprocessed":
        X_train, X_validation , X_test, target_train, target_validation, target_test, \
        Y_train , Y_validation, Y_test , classes_nb, classes_names = dataset.load_processed(target, undersampled == "undersampled")
    else:
        X_train, X_validation , X_test, target_train, target_validation, target_test, \
        Y_train , Y_validation, Y_test , classes_nb, classes_names =dataset.load_dataset(target, undersampled == "undersampled")

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_validation = np.reshape(X_validation, (X_validation.shape[0], X_validation.shape[1], 1))

    #we build the model
    if modelName == "CNN3":
        model = make_CNN_3layers(input_shape, classes_nb)
    elif modelName == "CNN8":
        model = make_CNN_8layers(input_shape, classes_nb)
    elif modelName == "CNNGRU":
        model = make_CNNGRU(input_shape, classes_nb)
    elif modelName == "CNNLSTM":
        model = make_CNNLSTM(input_shape, classes_nb)
    else:
        print("Bad model name")
        exit(-1)

    #string we will use to name the output files
    file_name = f"output/{modelName}_{target}_{undersampled}_{epoch}"

    #name of the file where we will save the best state achieved during training
    model_file = file_name + ".h5"

    callbacks = [
        learning_rate_schedulers[0],                                                # function updating learning rate
        ModelCheckpoint(model_file, save_best_only=True, monitor="val_loss"),       # save better model
        EarlyStopping(monitor="val_loss", patience=10, verbose=1)                   # stop training when no more learning
    ]

    model.compile(optimizer, loss_function, metrics=metrics)

    #training the model on dataset
    history = model.fit(X_train, Y_train, validation_data=(X_validation, Y_validation),
                       epochs=epoch, batch_size=batch_size, callbacks=callbacks)

    #we load the best state achieved during training
    model =keras.models.load_model(model_file)
    
    # we evaluate the best state model with the test dataset
    model.evaluate(X_test, Y_test, verbose =2)

    #we test the model and save its performance in a npy file
    save_in_dict(model, X_test, target_test, X_train, target_train, history, classes_nb, classes_names, file_name)