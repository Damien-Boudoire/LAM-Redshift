import sys
import keras
import numpy as np

from numpy.random import seed
import tensorflow as tf
from tensorflow.python.keras.backend import set_session

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

import dataset
from models import make_CNN_2inputs
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


if __name__ == '__main__':
    ### main function of this application
    #   usage : python main_CNN_2inputs.py target_name (full | undersampled) epoch batch
    #
    #   target_name : labels to use for classification
    #   undersampled : remove data to get balanced classes
    #   full : do not remove data
    #   epoch : number of epoch that the training will last
    #   batch : size of batch used
    ###

    try:
        target = sys.argv[1]
        if target not in ["DeltaZ", "Success2", "Success3", "Flag3"]:
            print(target)
            raise Exception
        undersample = sys.argv[2]
        if undersample not in ["undersampled", "full"]:
            print(undersample)
            raise Exception
        epoch = int(sys.argv[3])
        batch_size = int(sys.argv[4])
    except Exception:
        print("Bad arguments \n got : {0} \n need : target undersampled epoch batch".format(sys.argv))
        exit(-1)


    # we load the data
    X_train, X_validation, X_test, target_train, target_validation, target_test, \
        Y_train, Y_validation, Y_test, Redshifts_train, Redshifts_validation, Redshifts_test, \
        classes_nb, classes_names = dataset.load_processed_withRedshifts(target, undersample = undersample)


    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_validation = np.reshape(X_validation, (X_validation.shape[0], X_validation.shape[1], 1))

    # we build the model
    model = make_CNN_2inputs(input_shape, classes_nb)

    #string we will use to name the output files
    file_name = f"output/CNN_2inputs_{target}_{undersample}"

    #name of the file where we will save the best state achieved during training
    model_file = file_name + ".h5"

    callbacks = [
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=0.000001), # function updating learning rate
        ModelCheckpoint(model_file, save_best_only = True, monitor = "val_loss"),   # save better model
        EarlyStopping(monitor="val_loss", patience=5, verbose=1)                   # stop training when no more learning
    ]

    model.compile(optimizer, loss_function, metrics=metrics)

    #training the model on dataset
    history = model.fit([X_train, Redshifts_train], Y_train,
              epochs=epoch,
              batch_size=batch_size,
              callbacks=callbacks,
              verbose =1,
              validation_data = ([X_validation, Redshifts_validation], Y_validation))
    
    #we load the best state achieved during training
    model = keras.models.load_model(model_file)
    
    # we evaluate the best state model with the test dataset
    model.evaluate([X_test, Redshifts_test], Y_test, verbose =2)
    
    #we test the model and save its performance in a npy file
#    save_in_dict(model, X_test, target_test, X_train, target_train, history, classes_nb, classes_names, file_name)