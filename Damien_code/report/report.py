import json
import os
import numpy as np
import tensorflow as tf

from keras.callbacks import History as kHistory
from tensorflow.python.keras.callbacks import History as tfHistory
from keras.models import model_from_json
#from tensorflow.keras.utils import plot_model
from tensorflow.keras.backend import argmax
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report
from report.heatmap import heatmap, annotate_heatmap

path="/home/lam/Documents/redshift/report/"
#path="./report/"

def reportAsJSON(modelName, model, history, confusionMatrices,
                 classes, report):
    if isinstance(history, kHistory) or\
       isinstance(history, tfHistory) :
        history = history.history

    #rebuild history data casting keras.float32 object into float
    newHistory = {}
    for key, value in history.items():
        new_value = []
        for elmt in value:
            new_value.append(float(elmt))
        item = {key: new_value}
        newHistory.update(item)

    history = newHistory

    outPath = "{0}/{1}/".format(path, modelName)
    if not os.path.isdir(outPath):
        os.makedirs(outPath)

    report = {
        "name": modelName,
        "model": model.to_json(),
        "history": history,
        "confusion_train": confusionMatrices["train"].tolist(),
        "confusion_test": confusionMatrices["test"].tolist(),
        "classes": classes,
        "report": report
    }

    with open("{0}/report.json".format(outPath), "w") as fout:
        json.dump(report, fout)

def plotPerformance(modelName, history, confusionMatrices, classes, f1=None):
    outPath = "{0}/{1}/".format(path, modelName)
    if not os.path.isdir(outPath):
        os.makedirs(outPath)

    if isinstance(history, kHistory) or\
       isinstance(history, tfHistory) :
        history = history.history

    training_loss, validation_loss = history["loss"], history["val_loss"]
    if "accuracy"  in history:
        training_accuracy, validation_accuracy = history["accuracy"], history["val_accuracy"]
    else:
        training_accuracy, validation_accuracy = history["acc"], history["val_acc"]
    epoch_count = [*range(1, len(training_loss) + 1)]

    fig, ax = plt.subplots()

    ax.plot(epoch_count, training_loss, label="training loss")
    ax.plot(epoch_count, validation_loss, label="validation loss")

    ax.plot(epoch_count, training_accuracy, label="training accuracy")
    ax.plot(epoch_count, validation_accuracy, label="validation accuracy")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss / Accuracy")
    ax.set_title("Learning curves")
    ax.legend()
    fig.savefig("{}learning.png".format(outPath))

    matrixNames=["train", "test"]
    for cat in matrixNames:
        confusion = np.array(confusionMatrices[cat])
        fig, ax =plt.subplots()
        hm, _ = heatmap(confusion, classes, classes, ax=ax)
        annotate_heatmap(hm)
        fig.savefig("{0}confusion_{1}.png".format(outPath, cat))

def plotModel(modelName, model):
    outPath = "{0}/{1}/".format(path, modelName)
    if not os.path.isdir(outPath):
        os.makedirs(outPath)

    with open("{0}model.txt".format(outPath), "w") as fout:
        model.summary(print_fn=lambda s: fout.write(s+"\n"))

    #plot_model(model,"{0}ploted_model.png".format(outPath))

def reportFromJSON(filepath):
    file = open(filepath)
    report = json.load(file)
    model = model_from_json(report["model"])

    modelName = report["name"]

    if "accuracy" in report:
        accuracy = report["accuracy"]
        print(accuracy)
    if "f&" in report:
        f1_score = report["f1"]
        print(f1_score)

    plotModel(modelName, model)
    plotPerformance(modelName,\
                    report["history"],\
                    report["confusion"],\
                    report["classes"])


def testAndReport(modelName, model, classesName, history, x_train, y_train,x_test, y_test, ):
    y_pred = model.predict(x_test)
    y_fit = model.predict(x_train)
    if len(classesName) > 2:
        y_pred = argmax(y_pred)
        y_fit = argmax(y_fit)
    else:
        y_pred = y_pred.round().astype(int)
        y_fit = y_fit.round().astype(int)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))

    conf_train = confusion_matrix(y_train, y_fit)
    sumConf = conf_train.astype(np.float).sum(axis=1)

    conf_train = conf_train.astype(np.float)
    for i, s in enumerate(sumConf):
        conf_train[i] /= s
        conf_train[i] *=100


    conf_test = confusion_matrix(y_test, y_pred)
    sumConf = conf_test.astype(np.float).sum(axis=1)

    conf_test = conf_test.astype(np.float)
    for i, s in enumerate(sumConf):
        conf_test[i] /= s
        conf_test[i] *=100

    confusions = dict(train=conf_train, test=conf_test)
    reportAsJSON(modelName, model, history, confusions, classesName, report)
    plotPerformance(modelName, history, confusions, classesName)
