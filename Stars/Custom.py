import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot
from seaborn import heatmap

def Matrix(test:list, predict:list):
    pyplot.figure(figsize=(15, 8))
    heatmap(confusion_matrix(test, predict), annot=True)
    pyplot.ylabel("Prediction", fontsize=13)
    pyplot.xlabel("Actual", fontsize=13)
    pyplot.title("Confusion Matrix", fontsize=17)
    pyplot.show()

def Summary(Model_Name:str, test:list, predict:list):
    print(
        f"Classification Report for {Model_Name}:\n",
        classification_report(test, predict),
        "\n",
        f"F1 Score for {Model_Name}:",
        f1_score(test, predict, average="weighted"),
        "\n",
        f"Precision Score for {Model_Name}:",
        precision_score(test, predict, average="weighted"),
        "\n",
        f"Recall Score for {Model_Name}:",
        recall_score(test, predict, average="weighted"),
        "\n",
        f"Confusion Matrix for {Model_Name}:",
        confusion_matrix(test, predict),
    )

def All(Model_Name: str, test: list, predict: list):
    Summary(Model_Name, test, predict)
    Matrix(test, predict)

def Knn_Greedy(X_train: list,X_test: list,y_train: list,y_test: list,iter: bool = False,return_mode: int = 0,):
    """
    Bu fonksiyon, KNN algoritmasının en iyi k değerini bulmak için kendi yazdığım bir arama algoritmasıdır.
    Eğer veri setiniz çok büyük ve sistemiz güçlü değilse, bu fonksiyonu kullanmanızı önermem.
    Fonsiyon girdi olarak X_train, X_test, y_train, y_test alır ve en iyi k değerini ve skoru döndürür.
    Eğerki arkada kaçıncı döngüde olduğunu görmek istiyorsanız, iter değişkenini True yapabilirsiniz.
    Retun mode 0 ise sadece skoru döndürür, 1 ise sadece k değerini döndürür, 2 ise hem skoru hem k değerini döndürür.
    """
    score = 0
    for i in range(1, len(X_train), 2):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        _temp = knn.score(X_test, y_test)
        if _temp > score:
            score = _temp
            KNeighborsClassifier_Max_neighbors = i
        if iter:
            print(f"Score: {_temp} | K: {i}")
    if return_mode == 0:
        return score
    elif return_mode == 1:
        return KNeighborsClassifier_Max_neighbors
    elif return_mode == 2:
        return score, KNeighborsClassifier_Max_neighbors
    else:
        return "Error: return_mode must be 0, 1 or 2"

def Normalize(list: list):
    diff = max(list) - min(list)
    if diff == 0:
        return list
    for i in range(len(list)):
        x = list[i]
        x = (x - min()) / diff

def deNormalize():
    pass

