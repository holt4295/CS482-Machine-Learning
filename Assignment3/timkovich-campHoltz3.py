# -*- coding: utf-8 -*-
"""
Garrett Holtz
Sean Timkovich-Camp

Assignment 3
"""

"""
Imports
"""
from sklearn.linear_model import Ridge, Lasso, LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import math

"""
Global Declarations
"""
filePathReg = "machine.csv"
filePathClass = "haberman.csv"

"""
Import data from .csv to multiple numpy arrays
"""
def csvToNumPyArray(fileLocation):
    print("\nRunning csvToNumPyArray")
    targets = []
    with open(fileLocation) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row_num, row in enumerate(csv_reader):
            if row_num == 0:
                headers = row
            else:
                targets = np.append(targets, int(row[0]))
                
                    
    data = np.genfromtxt(fileLocation, delimiter=",", skip_header=1, dtype=float)
    np.set_printoptions(precision=3, suppress=True)
    
    print("\ndata-numpy array of shape ", data.shape)
    data = np.delete(data, 0, 1)
    
    print("target-numpy array of shape ", len(targets))
    
    target_name = headers.pop(0)
    #headers.pop() #Remove trailing header
    
    print("Target Name: ", target_name)
    print("Feature Names: ", headers)
    for index, row in enumerate(data):
        if index > 4:
            break
        print("Row ", index, ": ", row)
    return data, targets, headers
        
def correlateFeatures(data, headers):
    df = pd.DataFrame(data, columns=headers)
    corrDF = df.corr()
    print()
    print(corrDF,'\n')
    
def regModelFitting(data, targets, headers):
    #define the different alpha's to check
    parameters = {'alpha': [0, 0.1, 1, 10, 20, 50, 100]}
    
    #print(Ridge().get_params().keys())
    #print(Lasso().get_params().keys())
    #print(LinearRegression().get_params().keys())
    ridgeModel = Ridge()
    lassoModel = Lasso()
    ridgeGrid = GridSearchCV(ridgeModel, parameters, cv=5)
    lassoGrid = GridSearchCV(lassoModel, parameters, cv=5)
    linearRegModel = LinearRegression()
    X_train, X_test, Y_train, Y_test = train_test_split(data, targets, random_state=43 )
    #ridge100 = Ridge(alpha=0.1).fit(X_train, Y_train)
    ridgeClassifier = ridgeGrid.fit(X_train, Y_train)
    lassoClassifier = lassoGrid.fit(X_train, Y_train)
    linearRegClassifier = linearRegModel.fit(X_train, Y_train)
    
    print("Ridge~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Training Score: ", ridgeClassifier.score(X_train, Y_train))
    print("Test Score: ", ridgeClassifier.score(X_test, Y_test))
    print("Best Score: ", ridgeClassifier.best_score_)
    print("Best Parameters: ", ridgeClassifier.best_params_)
    print("Best estimator: ", ridgeClassifier.best_estimator_)
    print("r2: ", r2_score(Y_test, ridgeClassifier.predict(X_test)))
    print("RMSE: ", math.sqrt(mean_absolute_error(Y_test, ridgeClassifier.predict(X_test)))) 
    #print("Best estimator: ", ridgeClassifier.cv_results_)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Lasso~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Training Score: ", lassoClassifier.score(X_train, Y_train))
    print("Test Score: ", lassoClassifier.score(X_test, Y_test))
    print("Best Score: ", lassoClassifier.best_score_)
    print("Best Parameters: ", lassoClassifier.best_params_)
    print("Best estimator: ", lassoClassifier.best_estimator_)
    print("r2: ", r2_score(Y_test, lassoClassifier.predict(X_test)))
    print("RMSE: ", np.sqrt(mean_absolute_error(Y_test, lassoClassifier.predict(X_test)))) 
    #print("Best estimator: ", lassoClassifier.cv_results_)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Linear Regression~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Training Score: ", linearRegClassifier.score(X_train, Y_train))
    print("Test Score: ", linearRegClassifier.score(X_test, Y_test))
    print("r2: ", r2_score(Y_test, linearRegClassifier.predict(X_test)))
    print("RMSE: ", np.sqrt(mean_absolute_error(Y_test, linearRegClassifier.predict(X_test))))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    
    ridgeFig, ridgeAx = plt.subplots()
    ridgeAx.plot(Ridge(alpha=1).fit(X_train, Y_train).coef_, 's', label="Ridge alpha=1")
    ridgeAx.plot(Ridge(alpha=10).fit(X_train, Y_train).coef_, '^', label="Ridge alpha=10")
    ridgeAx.plot(Ridge(alpha=0.1).fit(X_train, Y_train).coef_, 'v', label="Ridge alpha=0.1")
    ridgeAx.plot(Ridge(alpha=100).fit(X_train, Y_train).coef_, 'v', label="Ridge alpha=100")
    ridgeAx.plot(linearRegClassifier.coef_, 'v', label="Linear Regression")
    ridgeAx.legend(ncol=2, loc=(0, 1.05))
    #ridgeAx.set_ylim(-1, 1)
    ridgeAx.set_xlabel("Coefficient index")
    ridgeAx.set_ylabel("Coefficient magnitude")
    
    lassoFig, lassoAx = plt.subplots()
    lassoAx.plot(Lasso(alpha=1).fit(X_train, Y_train).coef_, 's', label="Lasso alpha=1")
    lassoAx.plot(Lasso(alpha=10).fit(X_train, Y_train).coef_, '^', label="Lasso alpha=10")
    lassoAx.plot(Lasso(alpha=0.1).fit(X_train, Y_train).coef_, 'v', label="Lasso alpha=0.1")
    lassoAx.plot(Lasso(alpha=20).fit(X_train, Y_train).coef_, 'v', label="Lasso alpha=20")
    lassoAx.plot(linearRegClassifier.coef_, 'v', label="Linear Regression")
    lassoAx.legend(ncol=2, loc=(0, 1.05))
    #lassoAx.set_ylim(-1, 1)
    lassoAx.set_xlabel("Coefficient index")
    lassoAx.set_ylabel("Coefficient magnitude")
    

def classificationModelFitting(data, targets, headers):
       
    
    model = LogisticRegression()
    X_train, X_test, Y_train, Y_test = train_test_split(data, targets, test_size=0.2, shuffle=False)
    result = model.fit(X_train, Y_train)
    #print(result.predict_log_proba(X_train))
    print("Train Result: ", result.score(X_train, Y_train))
    print("Test Result: ", result.score(X_test, Y_test))
    
    y_pred_train = result.predict(X_train)
    con_matrix_train = confusion_matrix(Y_train, y_pred_train)
    
    y_pred_test = result.predict(X_test)
    con_matrix_test = confusion_matrix(Y_test, y_pred_test)
    
    print("\nConfusion Matrix Training:\n", con_matrix_train)
    print("Confusion Matrix Test:\n", con_matrix_test)
    
    print(classification_report(Y_test, y_pred_test, target_names=["> 5 years", "< 5 years"]))
    TP = con_matrix_test[0][0]
    FP = con_matrix_test[0][1]
    FN = con_matrix_test[1][0]
    TN = con_matrix_test[1][1]
    print(f"TP: {TP}\tFP: {FP}\tFN: {FN}\tTN: {TN}")
    print("Accuracy: ", (TP+TN)/(TP+TN+FP+FN))
    print("Precision: ", TP/(TP+FP))
    print("Recall: ", TP/(TP+FN))
    print("Specificity: ", TN/(TN+FP))
    print()
    
    #plotROC(Y_test, model.predict_log_proba(X_test)[::,1])
    plotROC(Y_test, model.decision_function(X_test))
    
def plotROC(y_test, y_pred_proba):
   
    temp = []
    for index, item in enumerate(y_test):
        if y_test[index] == 1:
            temp.append(0)
        else:
            temp.append(1)
    
    fpr, tpr, thresholds = roc_curve(temp, y_pred_proba)
    area = auc(fpr, tpr)
    
    gmeans = np.sqrt(tpr * (1 - fpr))
    opt_index = np.argmax(gmeans)
    close_zero = np.argmin(np.abs(thresholds))
    print("Best Threshold=%f, G-Mean=%.3f" % (thresholds[opt_index], gmeans[opt_index]))
    fig, ax = plt.subplots()
    ax.set_title("ROC Curve for Log Regression")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.plot(fpr, tpr, 'b', label = 'ROC Curve')
    #plt.scatter(fpr[opt_index], tpr[opt_index], marker='x', color='black', label="threshold zero")
    ax.plot(fpr[close_zero], tpr[close_zero], 'o', markersize=10, label="threshold zero", fillstyle="none", c='k', mew=2)
    ax.legend(loc = 'lower right')
    ax.plot([0,1], [0,1], 'r--')
    ax.set_xlim([0,1])
    
    return 0
    

reg_data, reg_targets, reg_headers = csvToNumPyArray(filePathReg)
correlateFeatures(reg_data, reg_headers)
regModelFitting(reg_data, reg_targets, reg_headers)

class_data, class_targets, class_headers = csvToNumPyArray(filePathClass)
correlateFeatures(class_data, class_headers)
classificationModelFitting(class_data, class_targets, class_headers)
