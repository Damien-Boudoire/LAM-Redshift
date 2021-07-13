# LAM-Redshift


Fichier "main.py"



Fichier "models.py"

# make_CNN
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

# make_CNNGru
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

# make_CNNLSTM
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


