from keras.models import Sequential
from keras.layers import Conv1D, GRU, LSTM, MaxPooling1D, Dense, Flatten

def make_CNN(input_shape, num_classes):

    model = Sequential()

    model.add(Conv1D(filters = 32, kernel_size = 9, activation = 'relu', input_shape = input_shape))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filters = 64, kernel_size = 6, activation = 'relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filters = 128, kernel_size = 3, activation = 'relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))

    model.add(Dense(32, activation='relu'))

    model.add(Dense(num_classes, activation='softmax'))

    return model


def makeCnnGRU(input, output, dropout=.2):
    model = Sequential()
    model.add(Conv1D(32, kernel_size=10, padding='same', activation='relu', input_shape=input))
    model.add(Conv1D(32, kernel_size=10, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(64, kernel_size=6, padding='same', activation='relu'))
    model.add(Conv1D(64, kernel_size=6, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(128, kernel_size=5, padding='same', activation='relu'))
    model.add(Conv1D(128, kernel_size=5, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(256, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv1D(256, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(GRU(512, return_sequences=True))
    model.add(GRU(256, return_sequences=True))
    model.add(GRU(128, return_sequences=True))
    model.add(GRU(64))
    model.add(Dropout(dropout))
    model.add(Flatten())

    # Output layer
    if output > 2:
        print("Model Categorical")
        model.add(Dense(units=output, activation="softmax"))
    else:
        print("Model Binary")
        model.add(Dense(units=1, activation="sigmoid"))
    return model


def makeCnnLSTM(nb_class):
    dim=(17908, 1)#,np.shape(X_train)[1:]
#    print(nb_class)

    if nb_class == 2:
      loss='binary_crossentropy'
      last_activation='Softmax'
    else :
      loss='categorical_crossentropy'
      last_activation='Softmax'

#    print(loss)

    model= Sequential()
    model.add(Conv1D(32,kernel_size=10,padding='same', activation='relu', input_shape=dim))
    model.add(Conv1D(32,kernel_size=10,padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(64,kernel_size=6,padding='same', activation='relu'))
    model.add(Conv1D(64,kernel_size=6,padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(128,kernel_size=5,padding='same',activation='relu'))
    model.add(Conv1D(128,kernel_size=5,padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(256,kernel_size=3,padding='same',activation='relu'))
    model.add(Conv1D(256,kernel_size=3,padding='same',activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(32))

    model.add(Dense(nb_class,activation=last_activation))

    return model