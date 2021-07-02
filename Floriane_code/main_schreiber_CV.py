import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
import numpy as np
from dataset_redshift import dataset_redshift
from LSTM_CNN_redshift import LSTM_CNN_redshift
from config_gpu_memory import config_gpu_memory

config_gpu_memory()

nom_class='DeltaZ'
Undersample=True
X_train, X_validation ,X_test, target_train, target_validation ,target_test , Y_train , Y_validation, Y_test , nb_class, nom_classes = dataset_redshift(nom_class,Undersample=Undersample)

X_test=np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))


inputs = np.concatenate((X_train, X_validation), axis=0)
targets = np.concatenate((Y_train, Y_validation), axis=0)

inputs=np.reshape(inputs, (inputs.shape[0], inputs.shape[1],1))

# Define the K-fold Cross Validator
num_folds=5
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
acc_per_fold=[]
loss_per_fold=[]


for train, valid in kfold.split(inputs, targets):
  print(targets[train] , targets[valid])

  # Define the model architecture
  model=LSTM_CNN_redshift()

  def scheduler(epoch, lr):
       if epoch < 10:
           return lr
       else:
           return lr * tf.math.exp(-0.1)

  reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {fold_no} ...')

  # Fit data to model
  history = model.fit(inputs[train], targets[train],
              batch_size=128,
              epochs=55,
              callbacks=[reduce_lr],
              verbose=1)


  # Generate generalization metrics
  scores = model.evaluate(inputs[valid], targets[valid], verbose=0)
  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_per_fold.append(scores[1] * 100)
  loss_per_fold.append(scores[0])

  # Increase fold number
  fold_no = fold_no + 1