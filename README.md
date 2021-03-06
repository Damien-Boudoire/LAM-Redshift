# LAM-Redshift


## Fichier "main.py"
Ce code est le script principal qui appelle toutes les autres fonctions. Il s'utilise de la manière suivante :  

python main.py model_name target_name (raw | preprocessed) (full | undersampled) epoch batch  

Les paramètres:
- model_name = CNN3 | CNN8 | CNNGRU | CNNLSTM  
    Le modèle à entrainer et tester 
- target_name = DeltaZ | Success2 | Success3 | Flag3  
     La classification choisie  
- raw : Charge les données depuis le fichier originel  
- preprocessed : utilise les données prédécoupées pour notre étude  
- undersampled : Rééquilibre les classes par sous-échantillonage  
- full : Garde l'ensemble des données sans rééquilibrage des classes  
- epoch : Nombre d'epochs max pour l'entrainement  
- batch : La taille des batchs  


## Fichier "dataset.py"

### load_processed
Ce code permet de charger les données déja pré-découpées en Train / Test / Validation

    load_processed(target, Undersample)
    
    Parameters
    ----------
    target : str 
        Choix de la classe : 'DeltaZ', 'Success2' ,'Flag3', 'Success3'
        'DeltaZ' : Classification en 2 classes selon le critère DeltaZ > 10^-3 et DeltaZ < 10^-3
        'Success2' : Classification en 2 classes 'Success' - 'Spurious/Mismatch' 
        'Flag3' : Classification en 3 classes 'Flag 1' , 'Flags 2&9' , 'Flag 3&4'
        'Success3' : Classification en 3 classes 'Success' - 'Spurious' - 'Mismatch'
    Undersample : bool
        True : Downsampling pour rééquilibrer les classes
        False : Pas de rééquilibrage
    
    Return
    -------
        X_train : array : Données pour la partie "Train"
        X_validation : Données pour la partie "Validation"
        X_test : Données pour la partie "Test"
        target_train : Labels pour la partie "Train"
        target_validation : Labels pour la partie "Validation"
        target_test : Labels pour la partie "Test"
        Y_train : Target en One-Hot-Encoding pour la partie "Train"
        Y_validation : Target en One-Hot-Encoding pour la partie "Validation"
        Y_test : Target en One-Hot-Encoding pour la partie "Test"
        nb_class : Nombre de classes
        nom_classes : Noms des différentes classes

### load_dataset
Ce code permet de charger les données depuis les fichiers d'origine (converties en 32bits), pdfs32.npy et attributes32.npy, et les préparent pour le modèle en 3 parties : "Train", "Test" et "Validation"

    load_processed(target, Undersample)
    
    Parameters
    ----------
    target : str 
        Choix de la classe : 'DeltaZ', 'Success2' ,'Flag3', 'Success3'
        'DeltaZ' : Classification en 2 classes selon le critère DeltaZ > 10^-3 et DeltaZ < 10^-3
        'Success2' : Classification en 2 classes 'Success' - 'Spurious/Mismatch' 
        'Flag3' : Classification en 3 classes 'Flag 1' , 'Flags 2&9' , 'Flag 3&4'
        'Success3' : Classification en 3 classes 'Success' - 'Spurious' - 'Mismatch'
    Undersample : bool
        True : Downsampling pour rééquilibrer les classes
        False : Pas de rééquilibrage
    
    Return
    -------
        X_train : array : Données pour la partie "Train"
        X_validation : Données pour la partie "Validation"
        X_test : Données pour la partie "Test"
        target_train : Labels pour la partie "Train"
        target_validation : Labels pour la partie "Validation"
        target_test : Labels pour la partie "Test"
        Y_train : Target en One-Hot-Encoding pour la partie "Train"
        Y_validation : Target en One-Hot-Encoding pour la partie "Validation"
        Y_test : Target en One-Hot-Encoding pour la partie "Test"
        nb_class : Nombre de classes
        nom_classes : Noms des différentes classes

## Fichier "models.py"

### make_CNN
Cette fonction crée le premier model de CNN de notre étude.
C'est un modèle simple mais qui donne déjà de bon résultat en 2 classes et a l'avantage d'être très rapide à apprendre.

    make_CNN(input_shape, num_classes)

    Parameters
    ----------
    input_shape : tuple
         Taille des données d'entrée du réseau de neurone,
         égal à (17908, 1) dans notre cas
    nb_class : int
        nombre de classes utilisées pour ce modèle, 
        correspond à la taille de la sortie du modèle

    Return
    -------
    model

### make_CNNGru
Cette fonction est le modèle CNN-Gru de notre étude.
    
    make_CNNGRU(input, output, dropout=.2)

    Parameters
    ----------
    input_shape : tuple
         Taille des données d'entrée du réseau de neurone,
         égal à (17908, 1) dans notre cas
    nb_class : int
        nombre de classes utilisées pour ce modèle, 
        correspond à la taille de la sortie du modèle

    Return
    -------
    model

### make_CNNLSTM
Cette fonction est le modèle CNN-LSTM de notre étude.
    
    make_CNNGRU(input, output)

    Parameters
    ----------
    input_shape : tuple
         Taille des données d'entrée du réseau de neurone,
         égal à (17908, 1) dans notre cas
    nb_class : int
        nombre de classes utilisées pour ce modèle, 
        correspond à la taille de la sortie du modèle

    Return
    -------
    model

## Fichier "main_CNN_2inputs.py"
Ce code est une version modifié du main pour tester le réseau CNN à 2 entrées.
Il s'utilise de la manière suivante:

python main_CNN_2inputs.py target_name (full | undersampled) epoch batch

Les paramètres:
- target_name : labels à utiliser pour la classification (DeltaZ | Success2 | Success3 | Flag3)  
- undersampled : utilise les données avec classes rééquilibrées (sous-échantillonées)  
- full : utilise l'ensemble des données  
- epoch : nombre d'epochs pour l'apprentissage  
- batch : taille des batchs  

## Report
Ce dossier sert pour l'affichage des résultats

### save_in_dict
Ce code sert à sauvegarder les résultats d'un modèle sous la forme d'un dictionnaire pour avoir accès a toutes les informations et les afficher avec fig.py

    save_in_dict(model, X_test, Y_test, X_train, Y_train, history, nb_class, nom_classes, file_name)
    
    Parameters
    ----------
    model : Sortie d'un des modèles (make_CNN...)
    X_test : Données de la partie "Test"
    Y_test : Target en OneHotEncoding de la partie "Test"
    X_train : Données de la partie "Train"
    Y_train : Target en OneHotEncoding de la partie "Train"
    history : history du modèle 
    nb_class : Nombres de classes
    nom_classes : Nom des classes
    file_name : Nom du fichier à sauvegarder
    
    Le dictionnaire contient les informations suivantes
    ----------
    "conf_matrix_train" : Matrice de confusion de la partie "Train"
    "conf_matrix_test" : Matrice de confusion de la partie "Test"
    "nb_class" : Nombres de classes
    "nom_classes" : Nom des classes
    "acc" : Valeur des accuracys lors de l'apprentissage
    "loss" : Valeur des loss lors de l'apprentissage
    "val_acc" : Valeur des validation accuracys lors de l'apprentissage
    "val_loss" : Valeur des validation loss lors de l'apprentissage
    "report" : Les différentes métriques (Précision, Rappel, F1-Score)
    "model_summary" : L'architecture du modèle
    "lr" : Valeurs du learning rate durant l'apprentissage


### fig
Ce code sert à afficher sous la forme d'un subplot, les 2 matrices de confusions ("Train" / "Test"), les courbes d'apprentissages (Accuracy, Loss, Validation Accuracy, Validation Loss) en fonction du nombre d'epochs et les différentes métriques de l'apprentissage

    

