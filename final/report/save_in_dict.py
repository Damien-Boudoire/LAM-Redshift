from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
####### Save en dictionnaire


def save_in_dict(model, X_test, Y_test, X_train, Y_train, history, class_nb, class_names, file_name):

    lr = history.history['lr']

    y_pred_test = model.predict(X_test)
    #y_pred_test = np.argmax(y_pred_test, axis=1)

    y_pred_train = model.predict(X_train)
    #y_pred_train = np.argmax(y_pred_train, axis=1)

#    if nb_class > 2:
    y_pred_test = np.argmax(y_pred_test, axis=1)
    y_pred_train = np.argmax(y_pred_train, axis=1)
#    else:
#    y_pred_test = y_pred_test.round().astype(int)
#    y_pred_train = y_pred_train.round().astype(int)

    conf_matrix_test = confusion_matrix(Y_test, y_pred_test)
    conf_matrix_train = confusion_matrix(Y_train, y_pred_train)

    report = classification_report(Y_test, y_pred_test, output_dict=True)

    acc = history.history['acc']
    loss = history.history['loss']
    val_acc = history.history['val_acc']
    val_loss = history.history['val_loss']

    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)

    dicte = {}
    dicte["conf_matrix_train"] = conf_matrix_train
    dicte["conf_matrix_test"] = conf_matrix_test
    dicte["nb_class"] = class_nb
    dicte["nom_classes"] = class_names
    dicte["acc"] = acc
    dicte["loss"] = loss
    dicte["val_acc"] = val_acc
    dicte["val_loss"] = val_loss
    dicte["report"] = report
    dicte["model_summary"] = short_model_summary
    dicte["lr"] = lr

    np.save(file_name, dicte)
