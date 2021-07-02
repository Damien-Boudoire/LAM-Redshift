import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from imblearn.under_sampling import RandomUnderSampler


path='../data/'

def dataset_redshift(target,size_test=0.15, input_data='logPdf', DeltaZ_limit=2e-3, Undersample=False):

    ##### Target #######
    # Les cibles peuvent être : 'Flag' , 'Success_2' , Success_3, 'Flag_class', 'DeltaZ'

#    pdf_zgrid=np.load(path+'zgrid32.npy')
    attributes=np.load(path+'attributes32.npy', allow_pickle=True)
    pdfs=np.load(path+'pdfs32.npy', allow_pickle=True)
    
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

        nom_classes= [f"$\Delta$Z $< {DeltaZ_limit}$", f"$\Delta$Z $>= {DeltaZ_limit}$"] #Arnaud: pour utilisation dans l'affichage de la matrice de confusion

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




    else:
        raise ValueError("Choisis la target = 'Flag' , 'Success' , 'Flag_class' ")


    nb_classes=len(np.unique(y))

    X_train, X_test, target_train, target_test = train_test_split(X, y, test_size=size_test)
    X_train, X_validation, target_train, target_validation = train_test_split(X_train, target_train, test_size=0.15)

    
    if Undersample :
        rus = RandomUnderSampler(random_state=0)
        X_train, target_train = rus.fit_resample(X_train, target_train)
        X_test, target_test = rus.fit_resample(X_test, target_test)
        X_validation, target_validation = rus.fit_resample(X_validation, target_validation)


    Y_train =  np_utils.to_categorical(target_train, nb_classes)   #convertir en one-hot-code
    Y_test = np_utils.to_categorical(target_test, nb_classes)
    Y_validation = np_utils.to_categorical(target_validation, nb_classes)

    

    return X_train, X_validation ,X_test, target_train, target_validation ,target_test , Y_train , Y_validation, Y_test , nb_classes, nom_classes


