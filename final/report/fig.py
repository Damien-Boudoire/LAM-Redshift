#from MLP_redshift import MLP_redshift
#from CNN1D_redshift import CNN1D_redshift
import sys
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import collections
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from imblearn.under_sampling import RandomUnderSampler
import seaborn as sns

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

file=sys.argv[1] #'BigCnnGRU_256_3_60_DeltaZ_True'

file = os.path.splitext(file)[0]

#best_epoch=25

read_dictionary = np.load(file+".npy",allow_pickle='TRUE').item()
nb_class=read_dictionary['nb_class']                  #read_dictionnary permet de lire dans le dictionnaire !! :-D
arr_test=read_dictionary['conf_matrix_test']
arr_train=read_dictionary['conf_matrix_train']
nom_classes = read_dictionary['nom_classes']


arr_test=arr_test.astype(float)
arr_train=arr_train.astype(float)

arr_test_pctge=np.zeros((nb_class,nb_class))
arr_train_pctge=np.zeros((nb_class,nb_class))


## Calcul des matrices en pourcentage stocké dans de nouvelles variables
for i in range(np.shape(arr_test_pctge)[1]):
  arr_test_pctge[i,:]=arr_test[i,:]/np.sum(arr_test[i,:])
  arr_train_pctge[i,:]=arr_train[i,:]/np.sum(arr_train[i,:])


# Création du subplot
fig, axs = plt.subplots(2 , 2 , figsize=(13,8) )

# Création des heatmaps
###############################################################################
#### Création de la "Heatmap test"
# Label pour afficher 2 valeurs dans la heatmap



group_counts = ['{0:0.0f}'.format(value) for value in
                arr_test.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     arr_test_pctge.flatten()]
labels = [f'{v1}\n\n{v2}' for v1, v2 in
          zip(group_percentages,group_counts)]

labels = np.asarray(labels).reshape(nb_class,nb_class)

# Affichage de la heatmap
hm=sns.heatmap(arr_test_pctge*100, annot=labels, fmt='', cmap='viridis', ax=axs[0, 1], linewidths=2,cbar=False)
hm.xaxis.tick_top() # x axis on top
hm.xaxis.set_label_position('top')
hm.set_xticklabels(nom_classes)
hm.set_yticklabels(nom_classes,va='center')
hm.set_title('(b) Test')


#### Création de la "Heatmap train"
# Label pour afficher 2 valeurs dans la heatmap
group_counts = ['{0:0.0f}'.format(value) for value in
                arr_train.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     arr_train_pctge.flatten()]
labels = [f'{v1}\n\n{v2}' for v1, v2 in
          zip(group_percentages,group_counts)]

labels = np.asarray(labels).reshape(nb_class,nb_class)


# Affichage de la heatmap
hm=sns.heatmap(arr_train_pctge*100, annot=labels, fmt='', cmap='viridis', ax=axs[0, 0], linewidths=2, cbar=False)
hm.xaxis.tick_top() # x axis on top
hm.xaxis.set_label_position('top')
hm.set_xticklabels(nom_classes)
hm.set_yticklabels(nom_classes,va='center')
hm.set_title('(a) Train')


#Affichage des learning curves
##########################################################################


acc=read_dictionary['acc']
loss=read_dictionary['loss']
val_acc=read_dictionary['val_acc']
val_loss=read_dictionary['val_loss']

max_value = np.max([np.max(acc), np.max(loss),
                    np.max(val_acc), np.max(val_loss)])

best_epoch = np.argmin(val_loss)+1

epoch=np.linspace(1,len(val_loss),len(val_loss)) # On redéfinit le nombre d'epochs en regardant la taille du vecteur de la val_loss

axs[1, 0].set_xlabel("Epoch")
axs[1, 0].set_ylabel("Accuracy / loss")
axs[1, 0].set_yticks(np.arange(0, 1.1, step=0.1))
axs[1, 0].set_ylim(0, 1)
#axs[1, 0].set_yticks(np.arange(0, np.round(max_value, 1)+0.1, step=0.1))   # A remodifier en fonction de la valeur max entre les 4 vecteurs

axs[1, 0].plot(epoch, acc, label='accuracy')
axs[1, 0].plot(epoch, val_acc, label='validation accuracy')

axs[1, 0].plot(epoch, loss, label='loss')
axs[1, 0].plot(epoch, val_loss, label='validation loss')
axs[1, 0].plot(np.ones(len(epoch))*best_epoch, np.linspace(0, np.round(max_value, 1)+0.1, len(epoch)), '--')

axs[1, 0].legend()
axs[1, 0].set_title("(c)")


##########################################################################


#Affichage du tableau
#################################################
report=read_dictionary['report']


cell_t=[] # On réccupère les différentes metrics pour chaque classe, elles sont sous forme de dictionnaire et on les mets dans une matrics pour l'affichage en tableau
for i in range(nb_class):
  cell=list(report.get(str(i)).values())
  cell_t.append(cell)

cell_text= np.round(cell_t,2)


columns= list(report['0'].keys()) # Colonne : 'Precision' 'Recall' 'F1-Score' 'Supports'
rows = nom_classes

print(rows)
print(report)

# Définition de la colormap du fond pour les lignes et les colonnes

colors = plt.cm.Reds(np.linspace(0.2, 0.5, len(rows)))
colors = colors[::-1]

colors2 = plt.cm.Reds(np.linspace(0.2, 0.5, len(columns)))
colors2 = colors2[::-1]


## Tableau

the_table = axs[1, 1].table(cellText=cell_text,
                      rowLabels=rows,
                      rowColours=colors,
                      colLabels=columns,
                      colColours=colors2,
                      bbox = [0.2, 0.1, 1, 0.8],
                      cellLoc='center')

the_table.set_fontsize = 30
#the_table.auto_set_column_width(col=1)
the_table.scale(1, 2)
axs[1, 1].axis('off')
axs[1,1].set_title("(d)")

plt.savefig(file+".pdf", format="pdf")
plt.show()
