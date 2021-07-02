from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
####### Save en dictionnaire


def Save_in_dict(model, X_test, target_test, X_train, target_train, h, nb_class, nom_classes, nom):
    y_pred_test=model.predict(X_test)			
    y_pred_test=np.argmax(y_pred_test, axis=1)

    conf_matrix_test=confusion_matrix(target_test, y_pred_test)     # Matrice de confusion "test"



    y_pred_train=model.predict(X_train)
    y_pred_train=np.argmax(y_pred_train, axis=1)

    conf_matrix_train=confusion_matrix(target_train, y_pred_train) # Matrice de confusion "train"


    report=classification_report(target_test, y_pred_test,output_dict=True) #output_dict=True permet d'avoir report sous la forme d'un dictionnaire pour extraire plus facilement les données


    acc=h.history['acc']			# h est le résultat du modèle entrainé ici
    loss=h.history['loss']
    val_acc=h.history['val_acc']
    val_loss=h.history['val_loss']


    ### Permet d'avoir le model summary sous forme de texte pour le conserver et le print
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    ###

    dicte={}
    dicte["conf_matrix_train"] = conf_matrix_train			# Matrice de confusion du train
    dicte["conf_matrix_test"] = conf_matrix_test			# Matrice de confusion du test
    dicte["nb_class"]=nb_class 								# Nombre de classes (en sortie de dataset_redshift)
    dicte["nom_classes"]=nom_classes						# Noms des classes (en sortie de dataset_redshift)
    dicte["acc"]=acc 										# Vecteurs des valeurs d'accuracy
    dicte["loss"]=loss 										# Vecteurs des valeurs de loss
    dicte["val_acc"]=val_acc								# Vecteurs des valeurs de validation accuracy
    dicte["val_loss"]=val_loss								# Vecteurs des valeurs de validation loss
    dicte["report"]=report 									# Dictionnaire des metriques dans report
    dicte["model_summary"]=short_model_summary 				# Le modèle

    np.save(nom, dicte)

