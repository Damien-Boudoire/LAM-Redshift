import sys
import os
import numpy as np

directoryPath = "data/"


def load_raw_data(target):
    ###
    #   load the pdfs, the labels corresponding to target and the rest of the attributes
    ###
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
    ###
    #   shuffle the three sets x, y and attributes respecting matching rows
    ###
    indexes = np.random.permutation(len(x))

    x = x[indexes]
    y = y[indexes]
    attributes = attributes[indexes]
    return x, y, attributes


def split_sets(x, y, attributes, test_ratio=.15, valid_ratio=.15):
    ###
    #   splits x, y and attributes into train, validation and test sets respecting given ratio and matching rows
    ###
    x, y, attributes = shuffle_sets(x, y, attributes)

    # calculate output sets' size
    test_size = int(len(x) * test_ratio)
    valid_size = int((len(x) - test_size) * valid_ratio)
    train_size = len(x) - test_size - valid_size

    # get the train set
    X_train = x[:train_size]
    Y_train = y[:train_size]
    att_train = attributes[:train_size]

    # get the validation set
    X_valid = x[train_size:train_size + valid_size]
    Y_valid = y[train_size:train_size + valid_size]
    att_valid = attributes[train_size:train_size + valid_size]

    # get the test set
    X_test = x[test_size:]
    Y_test = y[test_size:]
    att_test = attributes[test_size:]

    return X_train, Y_train, att_train,\
        X_valid, Y_valid, att_valid,\
        X_test, Y_test, att_test


def undersample(x, y, attributes):
    ###
    # build train, validation and test sets removing rows so the other classes
    # has a number of element equal to 110% of the smaller class
    ###
    shuffle_sets(x, y, attributes)
    nb_class = len(np.unique(y))
    # get elements by class
    class_indexes = [np.where(y == cl)[0] for cl in range(nb_class)]
    # find smaller class and calculate new size for other classes
    new_length = int(min([len(indexes) for indexes in class_indexes]) * 1.1)

    # resize the classes
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

    # split each resize class into train, validation and test set
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

    # regroup elements from all classes into final train, validation and test sets
    X_train = np.concatenate(X_train)
    Y_train = np.concatenate(Y_train)
    att_train = np.concatenate(att_train)

    X_valid = np.concatenate(X_valid)
    Y_valid = np.concatenate(Y_valid)
    att_valid = np.concatenate(att_valid)

    X_test = np.concatenate(X_test)
    Y_test = np.concatenate(Y_test)
    att_test = np.concatenate(att_test)

    # shuffle the sets
    X_train, Y_train, att_train = shuffle_sets(X_train, Y_train, att_train)
    X_valid, Y_valid, att_valid = shuffle_sets(X_valid, Y_valid, att_valid)
    X_test, Y_test, att_test = shuffle_sets(X_test, Y_test, att_test)

    return X_train, Y_train, att_train, X_valid, Y_valid, att_valid, X_test, Y_test, att_test


if __name__ == '__main__':
    ###
    #   script to generate train, validation and test sets into .npy files
    #   usage : python processData.py target to_balance
    #   target : the classification to label the pdfs must be in : DeltaZ, Success2, Success3, Flag3
    #   to_balance : if value is balanced the script will reduce the number of rows
    #                to limit size difference between classes
    #   the script will produce the following tree structure into the file system :
    #   data/target/to_balance/TRAIN/X.npy
    #                               /Y.npy
    #                               /attributes.npy
    #                         /VALID/X.npy
    #                               /Y.npy
    #                               /attributes.npy
    #                         /TEST/X.npy
    #                              /Y.npy
    #                              /attributes.npy
    ###

    # get parameters
    try:
        target = sys.argv[1]
        if target not in ["DeltaZ", "Success2", "Success3", "Flag3"]:
            raise Exception("first argument must be : DeltaZ, Success2, Success3 or Flag3")
        to_balance = sys.argv[2]
        if to_balance not in ["balanced", "unbalanced"]:
            raise Exception("second argument must be : balanced or unbalanced")
        x, y, attributes = load_raw_data(target)
    except Exception as e:
        print(e)
        exit(-1)

    if to_balance == "balanced":
        # if user ask for balanced set we use downsampling
        X_train, Y_train, att_train, X_valid, Y_valid, att_valid, X_test, Y_test, att_test = undersample(x, y, attributes)
    else:
        # if user ask for unbalanced sets we will use all the data
        X_train, Y_train, att_train, X_valid, Y_valid, att_valid, X_test, Y_test, att_test = split_sets(x, y, attributes)

    # define output directories
    outDirectory = f"{directoryPath}{target}/{to_balance}/"
    outTrain = f"{outDirectory}TRAIN/"
    outValid = f"{outDirectory}VALID/"
    outTest = f"{outDirectory}TEST/"

    # test and make if necessary the output directory for train set
    if not os.path.isdir(outTrain):
        os.makedirs(outTrain)
    # write the train set files
    np.save(f"{outTrain}X.npy", X_train, allow_pickle=True)
    np.save(f"{outTrain}Y.npy", Y_train, allow_pickle=True)
    np.save(f"{outTrain}attributes.npy", att_train, allow_pickle=True)

    # test and make if necessary the output directory for validation set
    if not os.path.isdir(outValid):
        os.makedirs(outValid)
    # write the validation set files
    np.save(f"{outValid}X.npy", X_valid, allow_pickle=True)
    np.save(f"{outValid}Y.npy", Y_valid, allow_pickle=True)
    np.save(f"{outValid}attributes.npy", att_valid, allow_pickle=True)

    # test and make if necessary the output directory for test set
    if not os.path.isdir(outTest):
        os.makedirs(outTest)
    # write the test set files
    np.save(f"{outTest}X.npy", X_test, allow_pickle=True)
    np.save(f"{outTest}Y.npy", Y_test, allow_pickle=True)
    np.save(f"{outTest}attributes.npy", att_test, allow_pickle=True)