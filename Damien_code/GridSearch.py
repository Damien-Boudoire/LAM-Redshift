import sys
from model.CnnGruModel import makeCnnGRUModel

from tensorflow.keras.optimizers import Adam
from dataset_redshift import dataset_redshift


from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

from keras.callbacks import LearningRateScheduler

import numpy as np
from numpy.random import seed

import tensorflow as tf
from tensorflow.python.keras.backend import set_session

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.compat.v1.Session(config=config))

seed(1)
tf.random.set_seed(1)

def setUpGruForGS(input, output, loss_fn, optimizer=Adam, learn_rate=.0001, epsilon=1e-8,units=128, depth=1, reduce=2, dropout=.2):
    gru = makeCnnGRUModel(units, input, output, depth, reduce, dropout)

    optimizer = optimizer(learning_rate=learn_rate, epsilon=epsilon)
    gru.compile(optimizer, loss_fn, metrics=["accuracy"])
    return gru

if __name__ == '__main__':

    if len(sys.argv) < 3:
        print("Need 2 parameters : target balanced")

    target = sys.argv[1]
    balanced = True if int(sys.argv[2]) == 1 else False

    X_train, X_validation, X_test, \
    target_train, target_validation, \
    target_test, Y_train, Y_validation, \
    Y_test, nb_classes, nom_classes = dataset_redshift(target, balanced)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_validation = np.reshape(X_validation, (X_validation.shape[0], X_validation.shape[1], 1))

    categorical = (nb_classes > 2)

    param_grid = dict(depth=[1, 3, 5],
                      reduce=[1, 2, 3])

    output = nb_classes
    input = X_train[0].shape

    loss_fn = "categorical_crossentropy" if categorical else "BinaryCrossentropy"
    param_grid.update(dict(input=[input], output=[output], loss_fn=[loss_fn]))

    model = KerasClassifier(build_fn=setUpGruForGS)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, verbose=3)

    X_train += X_validation
    Y_train += Y_validation


    indexes = range(len(X_train))
    np.random.shuffle(indexes)
    X_train = X_train[indexes]
    Y_train = Y_train[indexes]

    grid_result = grid.fit(X_train, Y_train)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
