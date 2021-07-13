import sys

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
tensorflow.random.set_seed(1)

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
    LearningRateScheduler(lambda epoch, lr: lr if epoch < 10 else lr * tf.math.exp(-0.1))]
###


if __name__ == '__main__':
    ### main function of this application
    #   usage : python main.py model_name target_name (raw | preprocessed) (full | undersampled) epoch batch
    #
    #   mode_name = CNN3 | CNN8 | CNNGRU | CNNLSTM
    #       the model to train and test
    #   target_name : labels to use for classification
    #   raw : build the dataset from the original files
    #   preprocessed : use the built datasets in data/
    #   undersampled : remove data to get balanced classes
    #   full : do not remove data
    #   epoch : number of epoch that the training will last
    #   batch : size of batch used
    ###

    try:
        modelName = sys.argv[1]
        preprocessed = sys.argv[2]
        target = sys.argv[3]
        undersampled = sys.argv[4]
        epoch = sys.argv[5]
        batch_size = sys.argv[6]
    except Exception as error:
        print(error)
        exit(-1)


    # we load the data using different function depending we are using preprocessed dataset or not
    if preprocessed == "preprocessed":
        X_train, X_validation ,X_test, target_train, target_validation, target_test,\
        Y_train , Y_validation, Y_test , nb_classes, nom_classes = dataset.load_processed(target, undersampled == "undersampled")
    else:
        X_train, X_validation ,X_test, target_train, target_validation, target_test,\
        Y_train , Y_validation, Y_test , nb_classes, nom_classes =dataset.load_dataset(target, undersampled == "undersampled")

    #we build the model
    if modelName == "CNN3":
        model = make_CNN_3layers(input_shape, nb_classes)
    elif modelName == "CNN8":
        model = make_CNN_8layers(input_shape, nb_classes)
    elif modelName == "CNNGRU":
        model = make_CNNGRU(input_shape, nb_classes)
    elif modelName == "CNNLSTM":
        model = makeCnnLSTM(input_shape, nb_classes)
    else:
        print("Bad model name")
        exit(-1)

    #string we will use to name the output files
    file_name = f"saved/{modelName}_{target}_{undersampled}_{epoch}"

    #name of the file where we will save the best state achieved during training
    model_file = file_name + ".h5"

    callbacks = [
        learning_rate_schedulers[0],                                                # function updating learning rate
        ModelCheckpoint(model_file, save_best_only = True, monitor = "val_loss"),   # save better model
        EarlyStopping(monitor="val_loss", patience=10, verbose=1)                   # stop training when no more learning
    ]

    model.compile(optimizer, loss_function, metrics=metrics)

    #training the model on dataset
    history =model.fit(X_train, Y_train, validation_data=(X_validation, Y_validation),
                       epoch=epoch, batch=batch_size, callbacks=callbacks)

    #we load the best state achieved during training
    model =keras.model.load_model(model_file)

    #we test the model and save its performance in a npy file
    Save_in_dict(model, X_test, target_test, X_train, target_train, history, nb_classes, nom_classes, file_name)