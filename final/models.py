from keras.models import Sequential, Model
from keras.layers import Conv1D, GRU, LSTM, MaxPooling1D, Dense, Flatten, Input, Concatenate


def make_CNN_3layers(input_shape, nb_class):
    """
    Cette fonction crée le premier model de CNN de notre étude.
    C'est un modèle simple mais qui donne déjà de bon résultats en 2 classes
    et a l'avantage d'être très rapide à apprendre.

    Parameters
    ----------
    input_shape : tuple
         Taille des données d'entrée du réseau de neurone,
         égal à (17908, 1) dans notre cas
    nb_class : int
        nombre de classes utilisées pour ce modèle, 
        correspond à la taille de la sortie du modèle

    Returns
    -------
    model : Le CNN 3 couches dans un model keras.

    """
    
    model = Sequential()

    model.add(Conv1D(filters = 32, kernel_size = 9, activation = 'relu', input_shape = input_shape))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filters = 64, kernel_size = 6, activation = 'relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filters = 128, kernel_size = 3, activation = 'relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))

    model.add(Dense(32, activation='relu'))

    model.add(Dense(nb_class, activation='softmax'))

    return model




def make_CNNGRU(input_shape, nb_class):
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

    model.add(GRU(512, return_sequences=True))
    model.add(GRU(256, return_sequences=True))
    model.add(GRU(128, return_sequences=True))
    model.add(GRU(64))

    model.add(Flatten())

    # Output layer
    model.add(Dense(units=nb_class, activation="softmax"))
    return model


def make_CNNLSTM(input_shape,nb_class):

    if nb_class == 2:
      loss='binary_crossentropy'
      last_activation='Softmax'
    else :
      loss='categorical_crossentropy'
      last_activation='Softmax'


    model= Sequential()
    model.add(Conv1D(32,kernel_size=10,padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv1D(32,kernel_size=10,padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(64,kernel_size=6,padding='same', activation='relu'))
    model.add(Conv1D(64,kernel_size=6,padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(128,kernel_size=5,padding='same',activation='relu'))
    model.add(Conv1D(128,kernel_size=5,padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(256,kernel_size=3,padding='same',activation='relu'))
    model.add(Conv1D(256,kernel_size=3,padding='same',activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(32))

    model.add(Dense(nb_class,activation=last_activation))

    return model



def make_CNN_8layers(input_shape, nb_class):
    """
    Cette fonction crée un CNN avec une architecture semblable aux CNN-RNN,
    en remplaçant les couches RNN par des couches Denses.
    Ce modèle nous a servi à valider l'intérêt des couches RNN par rapport aux Denses

    Parameters
    ----------
    input_shape : tuple
         Taille des données d'entrée du réseau de neurone,
         égal à (17908, 1) dans notre cas
    nb_class : int
        nombre de classes utilisées pour ce modèle, 
        correspond à la taille de la sortie du modèle

    Returns
    -------
    model : Le CNN 8 couches dans un model keras.

    """
    model= Sequential()
    model.add(Conv1D(32,kernel_size=10,padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv1D(32,kernel_size=10,padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2)) 

    model.add(Conv1D(64,kernel_size=6,padding='same', activation='relu'))
    model.add(Conv1D(64,kernel_size=6,padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))   

    model.add(Conv1D(128,kernel_size=5,padding='same',activation='relu'))
    model.add(Conv1D(128,kernel_size=5,padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(256,kernel_size=3,padding='same',activation='relu'))
    model.add(Conv1D(256,kernel_size=3,padding='same',activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Flatten())
    
    model.add(Dense(256))
    model.add(Dense(128))
    model.add(Dense(64))
    model.add(Dense(32))

    model.add(Dense(nb_class,activation='softmax'))

    return model



def make_CNN_2inputs(input_shape, nb_class):
    """
    Cette fonction crée un CNN 3 couches à 2 entrées.
    La première entrée est de la taille input_shape et correspond à la PDF ou logPDF,
    la deuxième entrée est un scalaire (float ou int) correspondant à la valeur du Redshift éstimé ou son indice dans zgrid.
    
    Cette architecture n'est qu'une piste d'étude, 
    elle n'a pour l'instant pas donnée de meilleur résultat que l'architecture à une seule entrée.

    Parameters
    ----------
    input_shape : tuple
         Taille des données d'entrée du réseau de neurone,
         égal à (17908, 1) dans notre cas
    nb_class : int
        nombre de classes utilisées pour ce modèle, 
        correspond à la taille de la sortie du modèle

    Returns
    -------
    model : Le CNN à 2 entrées dans un model keras.

    """    
    
    # Definition des 2 entrees
    Pdf = Input(shape=input_shape)
    Redshift = Input(shape=(1,))
    
    # CNN pour les Pdfs
    x = Conv1D(filters = 32, kernel_size = 9, activation = 'relu')(Pdf)
    x = MaxPooling1D(pool_size=2)(x)
    
    x = Conv1D(filters = 64, kernel_size = 6, activation = 'relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    x = Conv1D(filters = 128, kernel_size = 3, activation = 'relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    x = Flatten()(x)
    
    CNN_out = Dense(64, activation='relu')(x)

    # On concatene la sortie du CNN avec le Redshift d'entree
    merged = Concatenate()([CNN_out, Redshift])
    
    merged = Dense(32, activation='relu')(merged)
    
    output = Dense(nb_class, activation='softmax')(merged)
    
    # On definitit le model avec 2 entrées
    model = Model(inputs=[Pdf, Redshift], outputs=output)
    
    return model