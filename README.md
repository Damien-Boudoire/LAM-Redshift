# LAM-Redshift


## Fichier "main.py"



## Fichier "dataset.py"

### load_processed
Ce code permet de télécharger les données déja pré-découpées en Train / Test / Validation

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
    -------

### load_dataset
Ce code permet de télécharger toutes les données et les préparent pour le modèle en 3 parties : "Train", "Test" et "Validation"

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
    -------

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

    Return model
    -------

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

    Return model
    -------

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

    Return model
    -------


