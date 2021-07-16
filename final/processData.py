import sys
import os
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

    return x, np.asarray(y), attributes


def shuffle_sets(x, y, attributes):
    indexes = np.random.permutation(len(x))

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
    nb_class = len(np.unique(y))
    class_indexes = [np.where(y == cl)[0] for cl in range(nb_class)]
    new_length = int(min([len(indexes) for indexes in class_indexes]) * 1.1)

    resized_classes = [indexes[:min(new_length, len(indexes))] for indexes in class_indexes]

    X_train = []
    Y_train = []
    att_train = []
    X_valid = []
    Y_valid = []
    att_valid = []
    X_test  = []
    Y_test  = []
    att_test = []

    for resized in resized_classes:
        temp_X_train, temp_Y_train, temp_att_train, \
        temp_X_valid, temp_Y_valid, temp_att_valid, \
        temp_X_test, temp_Y_test, temp_att_test = split_sets(x[resized], y[resized], attributes[resized])

        X_train.append(temp_X_train)
        Y_train.append(temp_Y_train)
        att_train.append(temp_att_train)

        X_valid.append(temp_X_valid)
        Y_valid.append(temp_Y_valid)
        att_valid.append(temp_att_valid)

        X_test.append(temp_X_test)
        Y_test.append(temp_Y_test)
        att_test.append(temp_att_test)

    X_train = np.concatenate(X_train)
    Y_train = np.concatenate(Y_train)
    att_train = np.concatenate(att_train)

    X_valid = np.concatenate(X_valid)
    Y_valid = np.concatenate(Y_valid)
    att_valid = np.concatenate(att_valid)

    X_test = np.concatenate(X_test)
    Y_test = np.concatenate(Y_test)
    att_test = np.concatenate(att_test)

    X_train, Y_train, att_train = shuffle_sets(X_train, Y_train, att_train)
    X_valid, Y_valid, att_valid = shuffle_sets(X_valid, Y_valid, att_valid)
    X_test, Y_test, att_test = shuffle_sets(X_test, Y_test, att_test)

    return X_train, Y_train, att_train, X_valid, Y_valid, att_valid, X_test, Y_test, att_test


if __name__ == '__main__':
    try:
        target = sys.argv[1]
        to_balance = sys.argv[2]
        if to_balance not in ["balanced", "unbalanced"]:
            raise Exception("Bad parameter value")
        x, y, attributes = load_raw_data(target)
    except Exception as e:
        print(e)
        exit(-1)

    if to_balance == "balanced":
        X_train, Y_train, att_train, X_valid, Y_valid, att_valid, X_test, Y_test, att_test = undersample(x, y, attributes)
    else:
        X_train, Y_train, att_train, X_valid, Y_valid, att_valid, X_test, Y_test, att_test = split_sets(x, y, attributes)

    outDirectory = f"{directoryPath}{target}/{to_balance}/"
    outTrain = f"{outDirectory}TRAIN/"
    outValid = f"{outDirectory}VALID/"
    outTest = f"{outDirectory}TEST/"

    if not os.path.isdir(outTrain):
        os.makedirs(outTrain)
    np.save(f"{outTrain}X.npy", X_train, allow_pickle=True)
    np.save(f"{outTrain}Y.npy", Y_train, allow_pickle=True)
    np.save(f"{outTrain}attributes.npy", att_train, allow_pickle=True)

    if not os.path.isdir(outValid):
        os.makedirs(outValid)
    np.save(f"{outValid}X.npy", X_valid, allow_pickle=True)
    np.save(f"{outValid}Y.npy", Y_valid, allow_pickle=True)
    np.save(f"{outValid}attributes.npy", att_valid, allow_pickle=True)

    if not os.path.isdir(outTest):
        os.makedirs(outTest)
    np.save(f"{outTest}X.npy", X_test, allow_pickle=True)
    np.save(f"{outTest}Y.npy", Y_test, allow_pickle=True)
    np.save(f"{outTest}attributes.npy", att_test, allow_pickle=True)