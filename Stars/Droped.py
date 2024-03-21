import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot
from plotly import express as px
from plotly import graph_objects as go
from Custom import (
    Matrix, 
    Summary, 
    All, 
    Knn_Greedy,
    Normalize,
)
import time
import warnings
warnings.simplefilter(action='ignore', category=Warning)

def current_ms():
    return round(time.time() * 1000)

dataset = pd.read_csv('/home/emre/Desktop/gits/Machine-Learning-Practice/Stars/data.csv')
star_color_dict = {color: index for index, color in enumerate(dataset["Star color"].unique())}
star_class_dict = {s_class: index for index, s_class in enumerate(dataset["Spectral Class"].unique())}

dataset_replaced = dataset.copy()
dataset_replaced["Star color"] = dataset_replaced["Star color"].replace(star_color_dict)
dataset_replaced["Spectral Class"] = dataset_replaced["Spectral Class"].replace(star_class_dict)

dataset_lineer = dataset_replaced.copy()
final = []


for drop in range(len(dataset_replaced.columns)):
    ms = current_ms()   
    X = dataset_lineer.drop("Star color",axis=1)
    y = dataset_lineer["Star color"]
    if dataset_replaced.columns[drop] == "Star color":
        continue
    print(f"Droped: {dataset_replaced.columns[drop]}")
    X.drop(dataset_replaced.columns[drop], axis=1,inplace=True)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=30, random_state=42)

    from sklearn.linear_model import LinearRegression
    Lineer = LinearRegression()
    Lineer.fit(X_train, y_train)
    Lineer_score = Lineer.score(X_test, y_test)      

    from sklearn.linear_model import Ridge
    ridge = Ridge()
    ridge.fit(X_train, y_train)
    Ridge_score = ridge.score(X_test, y_test)

    from sklearn.linear_model import Lasso
    lasso = Lasso()
    lasso.fit(X_train, y_train)
    lasso_score = lasso.score(X_test, y_test)

    from sklearn.linear_model import LogisticRegression
    logistic_regression = LogisticRegression()
    logistic_regression.fit(X_train, y_train)
    LogisticRegression_Score = logistic_regression.score(X_test, y_test)

    from sklearn.neighbors import KNeighborsClassifier
    neibor = Knn_Greedy(X_train, X_test, y_train, y_test,False,1)
    knn = KNeighborsClassifier(n_neighbors=neibor)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    KNeighborsClassifier_Score = knn.score(X_test, y_test)
    
    from sklearn.svm import LinearSVC
    svm_Linear = LinearSVC()
    svm_Linear.fit(X_train, y_train)
    svm_Linear_score = svm_Linear.score(X_test, y_test)
    svm_Linear_predict = svm_Linear.predict(X_test)

    from sklearn.svm import SVC
    svm_rbf = SVC(kernel="rbf",cache_size=1024)
    svm_rbf.fit(X_train, y_train)
    svm_rbf_score = svm_rbf.score(X_test, y_test)
    svm_rbf_predict = svm_rbf.predict(X_test)

    from sklearn.svm import SVC
    svm_poly = SVC(kernel="poly",cache_size=1024)
    svm_poly.fit(X_train, y_train)
    svm_poly_score = svm_poly.score(X_test, y_test)
    svm_poly_predict = svm_poly.predict(X_test)

    from sklearn.tree import DecisionTreeClassifier
    decision_tree = DecisionTreeClassifier(criterion="entropy")
    decision_tree.fit(X_train, y_train)
    y_pred_dt = decision_tree.predict(X_test)
    decision_tree_accuracy = decision_tree.score(X_test, y_test)

    from sklearn.ensemble import RandomForestClassifier
    random_forest = RandomForestClassifier()
    random_forest.fit(X_train, y_train)
    random_forest_score = random_forest.score(X_test, y_test)
    random_forest_predict = random_forest.predict(X_test)
    cal_time = current_ms()-ms
    print("Calculation Done in {}miliseconds".format(cal_time))
    ## index 5 is None because its star color coralation value
    coral = [0.9758924349768133,
             1.3136189014260116,
             0.644445827334223,
             -3.110556724964831,
             1.2708388429965989,
             None,
             0.9818368950234468]
    ##scores = [Lineer_score,Ridge_score, lasso_score, LogisticRegression_Score, KNeighborsClassifier_Score, svm_Linear_score, svm_rbf_score , svm_poly_score, decision_tree_accuracy, random_forest_score]
    model_name = ['Linear Regression','Coralation', 'Ridge', 'Lasso', 'Logistic Regression', 'KNeighbors Classifier', 'SVC_Linear','SVC_Rbf','SVC_Poly', 'Decision Tree', 'Random Forest',"Time"]
    final_temp = [dataset_replaced.columns[drop],coral[drop], Lineer_score,Ridge_score, lasso_score, LogisticRegression_Score, KNeighborsClassifier_Score, svm_Linear_score, svm_poly_score , svm_poly_score, decision_tree_accuracy, random_forest_score,cal_time]
    final.append(final_temp)
    print("\n")
    

panda_final = pd.DataFrame(final, columns=["Droped","Coralation", 'Linear Regression', 'Ridge', 'Lasso', 'Logistic Regression', 'KNeighbors Classifier', 'SVC_Linear','SVC_Rbf','SVC_Poly', 'Decision Tree', 'Random Forest',"Time"])
panda_final.to_csv("Droped.csv", index=False)
print("All Done")
