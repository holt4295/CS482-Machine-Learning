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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.feature_selection import SelectPercentile, f_regression, SelectFromModel, SelectKBest, r_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
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
    print(np.arange(0, 2, 0.1))
    data, targets, headers = csvToNumPyArray(filePathReg)
    std_data = standardizeData(data)
    norm_data = normalizeData(data)
    std_norm_data = normalizeData(std_data)
    
    r_data, targets, r_headers = featureSelection(std_data, targets, headers)
    #split raw data into training and test
    printHeader("Raw data modeling")
    #xTrain, xTest, yTrain, yTest = train_test_split(data, targets, test_size = 0.2, random_state=42)
    xTrain, xTest, yTrain, yTest = train_test_split(r_data, targets, test_size = 0.2, random_state=42)
    
    xTrain = np.array(xTrain, dtype=np.float64)
    xTest = np.array(xTest, dtype=np.float64)
    yTrain = np.array(yTrain, dtype=np.float64)
    yTest = np.array(yTest, dtype=np.float64)
    
    fitLinearRegression(xTrain, xTest, yTrain, yTest)
    fitKNN(xTrain, xTest, yTrain, yTest)
    fitDecisionTree(xTrain, xTest, yTrain, yTest)
    fitNeuralNetwork(xTrain, xTest, yTrain, yTest)
    #fitGradientBoosting(xTrain, xTest, yTrain, yTest)
    
    printHeader("standardized data modeling")
    #split standardize data into training and test
    xTrain, xTest, yTrain, yTest = train_test_split(std_data, targets, test_size = 0.2, random_state=42)
    
    xTrain = np.array(xTrain, dtype=np.float64)
    xTest = np.array(xTest, dtype=np.float64)
    yTrain = np.array(yTrain, dtype=np.float64)
    yTest = np.array(yTest, dtype=np.float64)
    #print("Waka\n", xTrain[:5])
    
    fitLinearRegression(xTrain, xTest, yTrain, yTest)
    # fitKNN(xTrain, xTest, yTrain, yTest)
    # fitDecisionTree(xTrain, xTest, yTrain, yTest)
    # fitNeuralNetwork(xTrain, xTest, yTrain, yTest)
    # fitGradientBoosting(xTrain, xTest, yTrain, yTest)
    
    printHeader("normalized data modeling")
    #split normalized data into training and test
    xTrain, xTest, yTrain, yTest = train_test_split(norm_data, targets, test_size = 0.2, random_state=42)
    
    xTrain = np.array(xTrain, dtype=np.float64)
    xTest = np.array(xTest, dtype=np.float64)
    yTrain = np.array(yTrain, dtype=np.float64)
    yTest = np.array(yTest, dtype=np.float64)
    
    fitLinearRegression(xTrain, xTest, yTrain, yTest)
    # fitKNN(xTrain, xTest, yTrain, yTest)
    # fitDecisionTree(xTrain, xTest, yTrain, yTest)
    # fitNeuralNetwork(xTrain, xTest, yTrain, yTest)
    # fitGradientBoosting(xTrain, xTest, yTrain, yTest)
    
    # print("Testing with standardized and normalized data")
    # #split normalized data into training and test
    # xTrain, xTest, yTrain, yTest = train_test_split(std_norm_data, targets, test_size = 0.2, random_state=42)
    
    # xTrain = np.array(xTrain, dtype=np.float64)
    # xTest = np.array(xTest, dtype=np.float64)
    # yTrain = np.array(yTrain, dtype=np.float64)
    # yTest = np.array(yTest, dtype=np.float64)
    # # fitGradientBoosting(xTrain, xTest, yTrain, yTest)
    
    return 0


"""
Import data from .csv to multiple numpy arrays
"""
def csvToNumPyArray(fileLocation, preprocessData = True):
    # Preprocess the csv before reading
    if preprocessData:
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
                
    print("Target collected")
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
    print("Preprocessing data")
    data = pd.read_csv(fileLocation, header=0)

    # Update the file path with the processed csv
    outputFile = "processed_" + fileLocation
    
    # Convert Categorical data using one hot encoding
    data_dummies = pd.get_dummies(data, dtype=int)
    
    print(len(data_dummies.keys()))
    data_dummies.to_csv(outputFile, index= False)
    return outputFile

def standardizeData(data):
    print("Standardizing data")
    std_scaler = StandardScaler().fit(data)
    std_data = std_scaler.transform(data)
    
    return std_data

def normalizeData(data):
    print("Normalizing data")
    return normalize(data)

def featureSelection(data, targets, headers):
    print("Running feature selection")
    
    select = SelectPercentile(score_func=mutual_info_regression, percentile=50)
    
    #select = SelectFromModel(Ridge(), threshold = "median")

    selected_features = select.fit_transform(data, targets)
    selected_headers = select.get_feature_names_out(headers)

    mask = select.get_support()
    plt.matshow(mask.reshape(1,-1), cmap='gray_r')
    plt.xlabel("Select 70%")
    plt.yticks(())
    
    print("Finished feature selection")
    return selected_features, targets, selected_headers

def fitLinearRegression(xTrain, xTest, yTrain, yTest):
    print("Fitting Linear model")
    #lin_reg = LinearRegression().fit(xTrain, yTrain)
    
    parameters = {
        'alpha': np.arange(0.1, 3, step = 0.1)
        }
    
    lin_reg = GridSearchCV(Ridge(), parameters).fit(xTrain, yTrain)
    
    prediction = lin_reg.predict(xTest)
    print("Training: ", lin_reg.score(xTrain, yTrain))
    print("Test: ", lin_reg.score(xTest, yTest))
    print("RMSE: ", np.sqrt(mean_squared_error(yTest, prediction)))
    print("r2: ", r2_score(yTest, prediction))
    print("Finished fitting linear model")
    return 0

def fitGradientBoosting(xTrain, xTest, yTrain, yTest):
    print("Running Gradient Boosting Regressor")
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

def fitKNN(xTrain, xTest, yTrain, yTest):
    print("Fitting KNN model for regression")
    
    knn = KNeighborsRegressor(n_neighbors=3).fit(xTrain, yTrain)
    prediction = knn.predict(xTest)
    print("Scores 3KNN: ")
    print("Training: ", knn.score(xTrain, yTrain))
    print("Test: ", knn.score(xTest, yTest))
    print("RMSE: ", np.sqrt(mean_squared_error(yTest, prediction)))
    print("r2: ", r2_score(yTest, prediction))
    
    knn = KNeighborsRegressor(n_neighbors=5).fit(xTrain, yTrain)
    prediction = knn.predict(xTest)
    print("Scores 5KNN: ")
    print("Training: ", knn.score(xTrain, yTrain))
    print("Test: ", knn.score(xTest, yTest))
    print("RMSE: ", np.sqrt(mean_squared_error(yTest, prediction)))
    print("r2: ", r2_score(yTest, prediction))
    
    knn = KNeighborsRegressor(n_neighbors=7).fit(xTrain, yTrain)
    prediction = knn.predict(xTest)
    print("Scores 7KNN: ")
    print("Training: ", knn.score(xTrain, yTrain))
    print("Test: ", knn.score(xTest, yTest))
    print("RMSE: ", np.sqrt(mean_squared_error(yTest, prediction)))
    print("r2: ", r2_score(yTest, prediction))
    
    print("Finished fitting KNN model for regression\n")
    return 0

def fitDecisionTree(xTrain, xTest, yTrain, yTest):
    print("Fitting Decision Tree")
    d_tree = DecisionTreeRegressor(max_depth=10).fit(xTrain, yTrain)
    prediction = d_tree.predict(xTest)
    print("Scores Decision Tree: ")
    print("Training: ", d_tree.score(xTrain, yTrain))
    print("Test: ", d_tree.score(xTest, yTest))
    print("RMSE: ", np.sqrt(mean_squared_error(yTest, prediction)))
    print("r2: ", r2_score(yTest, prediction))
    
    print("Finished fitting Decision Tree\n")
    return 0

def fitNeuralNetwork(xTrain, xTest, yTrain, yTest):
    print("Fitting Neural Network")
    nn = MLPRegressor(random_state=42, max_iter=500).fit(xTrain, yTrain)
    prediction = nn.predict(xTest)
    print("Score: ", nn.score(xTest, yTest))
    print("RMSE: ", np.sqrt(mean_squared_error(yTest, prediction)))
    print("r2: ", r2_score(yTest, prediction))
    
    print("Finished fitting Neural Network\n")
    return 0

def printHeader(title):
    print()
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    print("\t\t\tStarting ", title, "\n")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    
main()