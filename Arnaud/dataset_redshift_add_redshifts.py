import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils


path='../data/'

def dataset_redshift(target,size_test, input_data='logPdf', DeltaZ_limit=3e-3):

    ##### Target #######
    # Les cibles peuvent être : 'Flag' , 'Success' , 'DeltaZ'

    pdf_zgrid=np.load(path+'zgrid32.npy')
    attributes=np.load(path+'attributes32.npy', allow_pickle=True)
    pdfs=np.load(path+'pdfs32.npy', allow_pickle=True)
    
    Redshifts = attributes[:,1].astype('float32')
    Redshift_indexes = np.array([np.where(pdf_zgrid == redshift)[0][0] for redshift in Redshifts]).astype('int32')
    
    ##### input_data #######
    # Les données d'entrées X peuvent être : 'logPdf', 'Pdf' et 'logVraisemblance'
        
    if input_data in ['logPdf', 'lP']:
        X = pdfs
    
    elif input_data in ['Pdf', 'P']:
        X = np.exp(pdfs)
    
    elif input_data in ['logVraisemblance', 'lV']:
        data_size = attributes.shape[0]
        logEvidence = attributes[:,6].astype(float)
        X = np.zeros(pdfs.shape)
        for i in range(data_size):
            X[i] = pdfs[i]+logEvidence[i]
    
    else:
        raise ValueError("input_data inconnue. input_data doit etre choisi parmis 'logPdf', 'Pdf' et 'logVraisemblance'")

                         
    if target == 'Flag':
        y=attributes[:,-1].astype(int)
        y=np.where(y==9, 0, y) 

        nom_classes = ["Flag 9", "Flag 1", "Flag 2", "Flag 3", "Flag 4"] #Arnaud: pour utilisation dans l'affichage de la matrice de confusion



    elif target == 'Success_2':
        flag=attributes[:,5]
        y=[]

        for i in flag:
            if i =='success':
                nw_fl=0
            else:
                nw_fl=1

            y.append(nw_fl)

        nom_classes = ["Success", "Spurious/missmatch"] #Arnaud: pour utilisation dans l'affichage de la matrice de confusion


    elif target == 'Success_3':
        flag=attributes[:,5]
        y=[]

        for i in flag:
            if i =='success':
                nw_fl=0
            elif i == 'spurious':
                nw_fl=1
            else:
                nw_fl=2

            y.append(nw_fl)

        nom_classes = ["Success", "Spurious" , "Missmatch"]


    elif target == 'Flag_class':
        flag=np.floor(attributes[:,-1])
        y=[]

        for i in flag:

            if (i == 2) or (i == 9):
                nw_fl=1
            elif (i == 3) or (i == 4):
                nw_fl=2
            elif i == 1:
                nw_fl=0

            y.append(nw_fl)

        nom_classes = ["flags 1", "flags 2-9", "flags 3-4"] #Arnaud: pour utilisation dans l'affichage de la matrice de confusion



    elif target == 'DeltaZ':
        flag=abs(attributes[:,3])
        y=[]

        for i in flag:

            if i < DeltaZ_limit:
                nw_fl=0
            elif i >= DeltaZ_limit:
                nw_fl=1

            y.append(nw_fl)

        nom_classes= [f"Deltaz<{DeltaZ_limit}", f"Deltaz>={DeltaZ_limit}"] #Arnaud: pour utilisation dans l'affichage de la matrice de confusion





    else:
        raise ValueError("Choisis la target = 'Flag' , 'Success' , 'Flag_class' ")


    nb_classes=len(np.unique(y))

    X_train, X_test, target_train, target_test, Redshift_indexes_train, Redshift_indexes_test = train_test_split(X, y, Redshift_indexes, test_size=size_test)


    Y_train =  np_utils.to_categorical(target_train, nb_classes)   #convertir en one-hot-code
    Y_test = np_utils.to_categorical(target_test, nb_classes)

    

    return X_train, X_test, target_train, target_test , Y_train , Y_test , Redshift_indexes_train, Redshift_indexes_test, nb_classes, nom_classes

