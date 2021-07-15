from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
####### Save en dictionnaire


def save_in_dict(model, X_test, Y_test, X_train, Y_train, history, class_nb, class_names, file_name):
    """
    Sauvegarde toutes les infos liées à l'apprentissage et aux performances du model dans un dictionnaire numpy.
    Ce dictionnaire est ensuite utiliser par fig.py pour afficher ces résultats.

    Parameters
    ----------
    model : model keras
        réseau de neurones entrainé
    X_test : tableau numpy à 2 dimensions
        Données de test
    Y_test : liste numpy
        targets de test
    X_train : tableau numpy à 2 dimensions
        Données de train
    Y_train : liste numpy
        target de train
    history : histrory keras
        historique de l'aprentissage du modèle
    class_nb : int
        Nombre de classe
    class_names : liste de string
        Noms des classes
    file_name : string
        Nom du fichier qui contiendra le dictionnaire

    """
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



def save_in_dict_2inputs(model, X_test, Redshifts_test, Y_test, X_train, Redshifts_train, Y_train, history, class_nb, class_names, file_name):
    """
    Meme fonction que save_in_dict mais permet de sauvegarder les infos d'un modèle à 2 entrées

    Parameters
    ----------
    model : model keras
        réseau de neurones entrainé
    X_test : tableau numpy à 2 dimensions
        Données de test
    Redshifts_test : int ou float
        indice ou valeur du Redshift de Test
    Y_test : liste numpy
        targets de test
    X_train : tableau numpy à 2 dimensions
        Données de train
    Redshifts_train : int ou float
        indice ou valeur du Redshift de Train
    Y_train : liste numpy
        target de train
    history : histrory keras
        historique de l'aprentissage du modèle
    class_nb : int
        Nombre de classe
    class_names : liste de string
        Noms des classes
    file_name : string
        Nom du fichier qui contiendra le dictionnaire

    """
    lr = history.history['lr']

    y_pred_test = model.predict([X_test, Redshifts_test])

    y_pred_train = model.predict([X_train, Redshifts_train])

    y_pred_test = np.argmax(y_pred_test, axis=1)
    y_pred_train = np.argmax(y_pred_train, axis=1)

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
