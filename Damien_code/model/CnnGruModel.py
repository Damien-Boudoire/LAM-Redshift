import math
from keras import Sequential
from keras.layers import GRU, Dropout, Dense, Conv1D, MaxPooling1D, Flatten

def makeCnnGRUModel(units, input, output, depth=2, reduce=2, dropout=.2):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=12, activation='relu', input_shape=input))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding="same"))

    model.add(Conv1D(filters=128, kernel_size=6, activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding="same"))

    model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding="same"))

    model.add(GRU(units=units, input_shape=input, return_sequences=(depth > 1)))
    model.add(Dropout(dropout))

    for i in range(1, depth):
        units = math.floor(units / reduce)
        model.add(GRU(units=units, return_sequences=(i < depth - 1)))
        model.add(Dropout(dropout))

    model.add(Dense(units=math.floor(units / reduce), activation="relu"))
    # Output layer
    if output > 2:
        print("Model Categorical")
        model.add(Dense(units=output, activation="softmax"))
    else:
        print("Model Binary")
        model.add(Dense(units=1, activation="sigmoid"))
    return model

def makeCnnGruModelAlt(input, output, dropout=.2):
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

    model.add(GRU(256, return_sequences=True))
    model.add(GRU(128, return_sequences=True))
    model.add(GRU(64, return_sequences=True))
    model.add(GRU(32))
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

def makeCnnGruBig(units, input, output, depth, reduce=2, dropout=.0):
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

    model.add(GRU(units=units, input_shape=input, return_sequences=(depth > 1)))
    model.add(Dropout(dropout))

    for i in range(1, depth):
        units = math.floor(units / reduce)
        model.add(GRU(units=units, return_sequences=(i < depth - 1)))
        model.add(Dropout(dropout))


    model.add(Dense(units=math.floor(units), activation="relu"))
    model.add(Dense(units=math.floor(output * 2), activation="relu"))

    # Output layer
    if output > 2:
        print("Model Categorical")
        model.add(Dense(units=output, activation="softmax"))
    else:
        print("Model Binary")
        model.add(Dense(units=1, activation="sigmoid"))
    return model