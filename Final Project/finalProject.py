# -*- coding: utf-8 -*-
"""
Garrett Holtz
Sean Timkovich-Camp
Cameron McCready

Final Project
"""

"""
Imports
"""
from sklearn.linear_model import Ridge, Lasso, LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.feature_selection import SelectPercentile,  mutual_info_regression, SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv

"""
Global Declarations
"""
filePathReg = "val_stats_cut.csv"

"""
Main function
"""
def main():
    data, targets, headers = csvToNumPyArray(filePathReg)
    std_data = standardizeData(data)
    
    #split data into training and test
    xTrain, xTest, yTrain, yTest = train_test_split(data, targets, test_size = 0.2, random_state=42)
    xTrain = np.array(xTrain, dtype=np.float64)
    xTest = np.array(xTest, dtype=np.float64)
    yTrain = np.array(yTrain, dtype=np.float64)
    yTest = np.array(yTest, dtype=np.float64)
    fitGradientBoosting(xTrain, xTest, yTrain, yTest)
    
    return 0


"""
Import data from .csv to multiple numpy arrays
"""
def csvToNumPyArray(fileLocation):
    # Preprocess the csv before reading
    fileLocation = preprocessCSV(fileLocation)
    
    print("\nRunning csvToNumPyArray")
    target_name = "win_percent"
    targets = []
    with open(fileLocation) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row_num, row in enumerate(csv_reader):
            if row_num == 0:
                headers = row
                #find target
                targetIndex = headers.index(target_name)
            else:
                targets = np.append(targets, row[targetIndex])
                
                    
    data = np.genfromtxt(fileLocation, delimiter=",", skip_header=1, dtype=float)
    np.set_printoptions(precision=3, suppress=True)
    
    print("\ndata-numpy array of shape ", data.shape)
    data = np.delete(data, targetIndex, 1)
    
    print("target-numpy array of shape ", len(targets))
    
    target_name = headers.pop(targetIndex) #Remove target header
    
    print("Target Name: ", target_name)
    print("Feature Names: ", headers)
    
    for index, row in enumerate(data):
        if index > 4:
            break
        print("Row ", index, ": ", row)
    #createHistograms(data, targets)
    
    return data, targets, headers

"""
Preprocess the csv file
"""
def preprocessCSV(fileLocation):
    data = pd.read_csv(fileLocation, header=0)

    # Update the file path with the processed csv
    outputFile = "processed_" + fileLocation
    
    # Convert Categorical data using one hot encoding
    data_dummies = pd.get_dummies(data, dtype=int)
    
    print(len(data_dummies.keys()))
    data_dummies.to_csv(outputFile, index= False)
    return outputFile

def standardizeData(data):
    std_scaler = StandardScaler().fit(data)
    std_data = std_scaler.transform(data)
    
    return std_data

def featureSelection(data, targets, headers):
    return 0

def fitGradientBoosting(xTrain, xTest, yTrain, yTest):
    params = {
        "n_estimators": 500,
        "max_depth": 4,
        "min_samples_split": 5,
        "learning_rate": 0.01,
        "loss": "squared_error",
     }
    reg = GradientBoostingRegressor(**params)
    reg.fit(xTrain, yTrain)
    
    mse = mean_squared_error(yTest, reg.predict(xTest), squared=False)
    print("The root mean squared error (RMSE) on test set: {:.4f}".format(mse))
    
    test_score = np.zeros((params["n_estimators"],), dtype=np.float64)
    for i, y_pred in enumerate(reg.staged_predict(xTest)):
        test_score[i] = mean_squared_error(yTest, y_pred)
    
    fig = plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.title("Deviance")
    plt.plot(
        np.arange(params["n_estimators"]) + 1,
        reg.train_score_,
        "b-",
        label="Training Set Deviance",
    )
    plt.plot(
        np.arange(params["n_estimators"]) + 1, test_score, "r-", label="Test Set Deviance"
    )
    plt.legend(loc="upper right")
    plt.xlabel("Boosting Iterations")
    plt.ylabel("Deviance")
    fig.tight_layout()
    plt.show()
    return 0

main()