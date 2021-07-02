# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 16:20:23 2021

@author: Arnaud
"""

import matplotlib.pyplot as plt

def plot_learning_curves(history):
    
    #print history.history.keys()
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'])
    plt.savefig('courbe_apprentissage_acc.pdf')
    
    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'])
    plt.savefig('courbe_apprentissage_loss.pdf')
    
    # summarize history for learning rate
    plt.figure()
    plt.plot(history.history['lr'])
    plt.title('model learning rate')
    plt.ylabel('learning rate')
    plt.xlabel('epoch')
    plt.savefig('courbe_apprentissage_lr.pdf')