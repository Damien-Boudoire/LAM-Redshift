import sys
import numpy as np

directoryPath = "data/"


def load_raw_data(target):
    x = np.load(f"{directoryPath}pdfs32.npy", allow_pickle=True)
    attributes = np.load(f"{directoryPath}attributes32.npy", allow_pickle=True)

    if target == 'Flag':
        y = np.floor(attributes[:, -1]).astype(int)
        y = np.where(y == 9, 0, y)

        nom_classes = ["Flag 9", "Flag 1", "Flag 2", "Flag 3", "Flag 4"]

    elif target == 'Success2':
        flag = attributes[:, 5]
        y = []

        for i in flag:
            if i == 'success':
                nw_fl = 0
            else:
                nw_fl = 1

            y.append(nw_fl)

        nom_classes = ["Success", "Spurious/Mismatch"]

    elif target == 'Flag3':
        flag = np.floor(attributes[:, -1]).astype(int)
        y = []

        for i in flag:

            if (i == 2) or (i == 9):
                nw_fl = 1
            elif (i == 3) or (i == 4):
                nw_fl = 2
            elif i == 1:
                nw_fl = 0

            y.append(nw_fl)

        nom_classes = ["Flag 1", "Flags 2-9", "Flags 3-4"]

    elif target == 'DeltaZ':
        flag = attributes[:, 3]

        y = []

        flag = abs(flag)

        for i in flag:

            if i < 5e-3:
                nw_fl = 0
            elif i >= 5e-3:
                nw_fl = 1

            y.append(nw_fl)

        nom_classes = ["Deltaz<1e-3", "Deltaz>=1e-3"]

    elif target == 'Success3':
        flag = attributes[:, 5]
        y = []

        for i in flag:
            if i == 'success':
                nw_fl = 0
            elif i == 'spurious':
                nw_fl = 1
            else:
                nw_fl = 2

            y.append(nw_fl)

        nom_classes = ["Success", "Spurious", "Mismatch"]

    else:
        raise ValueError("Choisis la target = 'Flag' , 'Success2' , 'Flag3' , 'DeltaZ' , 'Success3' ")

    return x, y, attributes


def shuffle_sets(x, y, attributes):
    indexes = np.arange(len(x))
    np.random.shuffle(indexes)

    x = x[indexes]
    y = y[indexes]
    attributes = attributes[indexes]
    return x, y, attributes


def split_sets(x, y, attributes, test_ratio=.15, valid_ratio=.15):

    x, y, attributes = shuffle_sets(x, y, attributes)

    test_size = int(len(x) * test_ratio)
    valid_size = int((len(x) - test_size) * valid_ratio)
    train_size = len(x) - test_size - valid_size
    X_train = x[:train_size]
    Y_train = y[:train_size]
    att_train = attributes[:train_size]

    X_valid = x[train_size:train_size + valid_size]
    Y_valid = y[train_size:train_size + valid_size]
    att_valid = attributes[train_size:train_size + valid_size]

    X_test = x[test_size:]
    Y_test = y[test_size:]
    att_test = attributes[test_size:]

    return X_train, Y_train, att_train,\
        X_valid, Y_valid, att_valid,\
        X_test, Y_test, att_test


def undersample(x, y, attributes):
    shuffle_sets(x, y, attributes)
    nb_class = np.unique(y)
    class_indexes = [np.where(y == cl)[0] for cl in range(nb_class)]
    class_min = min(enumerate(class_indexes), key=lambda x: len(x[1]))
    min_length = len(class_indexes[class_min])

    resized_classes = [indexes[:min_length] for indexes in class_indexes]

    X_train = Y_train = att_train = []
    X_valid = Y_valid = att_valid = []
    X_test  = Y_test  = att_test = []

    for resized in resized_classes:
        temp_X_train, temp_Y_train, temp_att_train, \
        temp_X_valid, temp_Y_valid, temp_att_valid, \
        temp_X_test, temp_Y_test, temp_att_test = split_sets(x[resized], y[resized], attributes[resized])

        X_train += temp_X_test
        Y_train += temp_Y_train
        att_train += temp_att_train

        X_valid += temp_X_valid
        Y_valid += temp_Y_valid
        att_valid += temp_att_valid

        X_test += temp_X_test
        Y_test += temp_Y_test
        att_test += temp_att_test

    return np.asarray(X_train), np.asarray(Y_train), np.asarray(att_train), \
           np.asarray(X_valid), np.asarray(Y_valid), np.asarray(att_valid), \
           np.asarray(X_test), np.asarray(Y_test), np.asarray(att_test),


if __name__ == '__main__':
    try:
        target = sys.argv[1]
        to_balance = sys.argv[2]
        x, y, attributes = load_raw_data(target)
    except Exception as e:
        print(e)
        exit(-1)

    if to_balance == "balance"
        X_train, Y_train, att_train, X_valid, Y_valid, att_valid, X_test, Y_test, att_test = split_sets(x, y, attributes)
    else:
        X_train, Y_train, att_train, X_valid, Y_valid, att_valid, X_test, Y_test, att_test = undersample(x, y, attributes)

    outDirectory = f"{directoryPath}{target}/{to_balance}/"
    outTrain = f"{outDirectory}TRAIN/"
    outValid = f"{outDirectory}VALID/"
    outTest = f"{outDirectory}TEST/"

    np.save(f"{outTrain}X.npy", X_train, allow_pickle=True)
    np.save(f"{outTrain}Y.npy", Y_train, allow_pickle=True)
    np.save(f"{outTrain}attributes.npy", att_train, allow_pickle=True)

    np.save(f"{outValid}X.npy", X_valid, allow_pickle=True)
    np.save(f"{outValid}Y.npy", Y_valid, allow_pickle=True)
    np.save(f"{outValid}attributes.npy", att_valid, allow_pickle=True)

    np.save(f"{outTest}X.npy", X_test, allow_pickle=True)
    np.save(f"{outTest}Y.npy", Y_test, allow_pickle=True)
    np.save(f"{outTest}attributes.npy", att_test, allow_pickle=True)