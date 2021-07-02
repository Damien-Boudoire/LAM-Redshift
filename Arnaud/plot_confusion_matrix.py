# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 16:57:41 2021

@author: Arnaud
"""

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import numpy as np
import matplotlib.pyplot as plt

def plot_CM(model, X, target, nom_classes, nom_img):
    Y_pred=model.predict(X)
    y_pred=np.argmax(Y_pred, axis=1)
    
    f1 = f1_score(target, y_pred, average=None)
    print("F1 score: ", f1)
    
    cm = confusion_matrix(target, y_pred)
    total = np.sum(cm, axis=1)
    cm_p = [cm[i]/total[i] for i in range(cm.shape[0])]
    cm_p = np.asarray(cm_p)*100
    print(cm_p)
    
    
    plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_p, display_labels=nom_classes)
    disp.plot()
#    plt.title("F1 score = " + f1)
    plt.savefig(nom_img)
    
def plot_CM_Redshift(model, X, Redshift, target, nom_classes, nom_img):
    Y_pred=model.predict([X, Redshift])
    y_pred=np.argmax(Y_pred, axis=1)
    
    f1 = f1_score(target, y_pred, average=None)
    print("F1 score: ", f1)
    
    cm = confusion_matrix(target, y_pred)
    print(cm)
    plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nom_classes)
    disp.plot()
    plt.savefig(nom_img)