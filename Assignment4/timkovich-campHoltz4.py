# -*- coding: utf-8 -*-
"""
Garrett Holtz
Sean Timkovich-Camp

Assignment 4
"""

"""
Imports
"""
from sklearn.linear_model import Ridge, Lasso, LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.feature_selection import SelectPercentile,  mutual_info_regression, SelectFromModel
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import math

"""
Global Declarations
"""
filePathReg = "houseSalePrices.csv"

"""
Import data from .csv to multiple numpy arrays
"""
def csvToNumPyArray(fileLocation):
    print("\nRunning csvToNumPyArray")
    target_name = "SalePrice"
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
    createHistograms(data, targets)
    
    return data, targets, headers

"""
Create Histograms based off features 1 and 2
"""
def createHistograms(data, targets):
    print("\nRunning createHistograms")
    lotAreaData = np.array([])
    overallQualityData = np.array([])
    salesPriceData = np.array([])
    for index, row in enumerate(data):
        lotAreaData = np.append(lotAreaData, data[index, 2])
        overallQualityData = np.append(overallQualityData, data[index, 3])
        salesPriceData = np.append(overallQualityData, targets[index])
        
    #Radius1 Hist
    fig1, ax1 = plt.subplots()

    fig1.suptitle("Lot area Data Histogram")
    ax1.set_xlabel("Square footage")
    ax1.set_ylabel("Frequency")
    ax1.hist(lotAreaData, facecolor="blue",  bins=200)
    
    #Texture1 Hist
    fig2, ax2 = plt.subplots()
    
    fig2.suptitle("House Quality Data Histogram")
    ax2.set_xlabel("Quality out of 10")
    ax2.set_ylabel("Frequency")
    ax2.hist(overallQualityData, facecolor="blue", edgecolor="gray", bins=10)
    

"""
Preprocess the data
"""
def preprocessCSV(fileLocation):
    data = pd.read_csv(fileLocation, header=0)
    del data['Id']
    """
    print(data.keys())
    #print(data.apply(pd.Series.value_counts))
    print(data.MSZoning.value_counts(dropna=False))
    print(data.Alley.value_counts(dropna=False))
    print(data.MasVnrType.value_counts(dropna=False))
    print(data.MasVnrArea.value_counts(dropna=False))
    print(data.BsmtQual.value_counts(dropna=False))
    print(data.BsmtCond.value_counts(dropna=False))
    print(data.BsmtExposure.value_counts(dropna=False))
    print(data.BsmtFinType1.value_counts(dropna=False))
    print(data.BsmtFinSF1.value_counts(dropna=False))
    print(data.BsmtFinType2.value_counts(dropna=False))
    print(data.BsmtFinSF2.value_counts(dropna=False))
    print(data.BsmtUnfSF.value_counts(dropna=False))
    print(data.Electrical.value_counts(dropna=False))
    print(data.LowQualFinSF.value_counts(dropna=False))
    print(data.GarageType.value_counts(dropna=False))
    print(data.GarageYrBlt.value_counts(dropna=False))
    print(data.GarageFinish.value_counts(dropna=False))
    print(data.GarageQual.value_counts(dropna=False))
    print(data.PoolQC.value_counts(dropna=False))
    print(data.Fence.value_counts(dropna=False))
    print(data.MiscFeature.value_counts(dropna=False))
    print(data.MSSubClass.value_counts(dropna=False))
    
    print(data.LotFrontage.value_counts(dropna=False))
    print(data.LotArea.value_counts(dropna=False))
    print(data.Street.value_counts(dropna=False))
    
    print(data.LotShape.value_counts(dropna=False))
    print(data.LandContour.value_counts(dropna=False))
    print(data.Utilities.value_counts(dropna=False))
    print(data.LotConfig.value_counts(dropna=False))
    print(data.LandSlope.value_counts(dropna=False))
    print(data.Condition1.value_counts(dropna=False))
    print(data.Condition2.value_counts(dropna=False))
    print(data.BldgType.value_counts(dropna=False))
    print(data.HouseStyle.value_counts(dropna=False))
    print(data.OverallQual.value_counts(dropna=False))
    print(data.OverallCond.value_counts(dropna=False))
    print(data.YearBuilt.value_counts(dropna=False))
    print(data.YearRemodAdd.value_counts(dropna=False))
    print(data.RoofStyle.value_counts(dropna=False))
    print(data.RoofMatl.value_counts(dropna=False))
    print(data.Exterior1st.value_counts(dropna=False))
    print(data.Exterior2nd.value_counts(dropna=False))
   
    print(data.ExterQual.value_counts(dropna=False))
    print(data.ExterCond.value_counts(dropna=False))
    print(data.Foundation.value_counts(dropna=False))
    
    print(data.TotalBsmtSF.value_counts(dropna=False))
    print(data.Heating.value_counts(dropna=False))
    print(data.HeatingQC.value_counts(dropna=False))
    print(data.CentralAir.value_counts(dropna=False))

    #print(data.1stFlrSF.value_counts(dropna=False))
    #print(data.2ndFlrSF.value_counts(dropna=False))
    
    print(data.GrLivArea.value_counts(dropna=False))
    print(data.BsmtFullBath.value_counts(dropna=False))
    print(data.BsmtHalfBath.value_counts(dropna=False))
    print(data.FullBath.value_counts(dropna=False))
    print(data.HalfBath.value_counts(dropna=False))
    print(data.BedroomAbvGr.value_counts(dropna=False))
    print(data.KitchenAbvGr.value_counts(dropna=False))
    print(data.KitchenQual.value_counts(dropna=False))
    print(data.TotRmsAbvGrd.value_counts(dropna=False))
    print(data.Functional.value_counts(dropna=False))
    print(data.Fireplaces.value_counts(dropna=False))
    print(data.FireplaceQu.value_counts(dropna=False))
    
    print(data.GarageCars.value_counts(dropna=False))
    
    print(data.GarageArea.value_counts(dropna=False))
   
    print(data.GarageCond.value_counts(dropna=False))
    print(data.PavedDrive.value_counts(dropna=False))
    print(data.WoodDeckSF.value_counts(dropna=False))
    print(data.OpenPorchSF.value_counts(dropna=False))
    print(data.EnclosedPorch.value_counts(dropna=False))
    #print(data.3SsnPorch.value_counts(dropna=False))
    print(data.ScreenPorch.value_counts(dropna=False))
    print(data.PoolArea.value_counts(dropna=False))
   
    
    
    print(data.MiscVal.value_counts(dropna=False))
    print(data.MoSold.value_counts(dropna=False))
    print(data.YrSold.value_counts(dropna=False))
    print(data.SaleType.value_counts(dropna=False))
    print(data.SaleCondition.value_counts(dropna=False))
    """
    
    outputFile = "processed_houseSalePrices.csv"
    values = {
        "LotFrontage":  0,
        "Alley":        "None",
        "MasVnrType":   "None",
        "MasVnrArea":   0,
        "BsmtQual":     "NB",
        "BsmtCond":     "NB",
        "BsmtExposure": "No",
        "BsmtFinType1": "NoB",
        "BsmtFinType2": "NoB",
        "Electrical":   "NoEle",
        "GarageType":   "DrvWay",
        "GarageFinish": "Unf",
        "GarageQual":   "OP",
        "GarageCond":   "OP",
        "PoolQC":       "NP",
        "Fence":        "NF",
        "MiscFeature":  "None"
    }
    data = data.fillna(value=values)
    data = data.ffill()
    
    data_dummies = pd.get_dummies(data, dtype=int)
    
    print(len(data_dummies.keys()))
    data_dummies.to_csv(outputFile, index= False)
    return outputFile

def reduceFeatures(data, targets, headers):
    print("\n\n")
    #split the data
    xTrain, xTest, yTrain, yTest = train_test_split(data, targets, random_state=42)
    #print(xTrain)
    select = SelectPercentile(score_func=mutual_info_regression, percentile=70)
    xTrain_selected = select.fit_transform(xTrain, yTrain)
    xTest_selected = select.transform(xTest)
    
    headers_selected = select.get_feature_names_out(headers)
    mask = select.get_support()
    plt.matshow(mask.reshape(1,-1), cmap='gray_r')
    plt.xlabel("Select 70%")
    plt.yticks(())
    #print(headers_selected)
    print("Remaining Feature #: ",len(xTrain_selected[0]))
    print('Univariate Correlation:\n', select.scores_)
    
    #Model Reduction
    xTrain, xTest, yTrain, yTest, headers_selected = modelReduction(xTrain_selected, xTest_selected, yTrain, yTest, headers_selected)
    
    return xTrain, xTest, yTrain, yTest, headers_selected

def modelReduction(xTrain, xTest, yTrain, yTest, headers):
    select = SelectFromModel(estimator= LinearRegression(), threshold="median").fit(xTrain, yTrain)
    xTrain_trans = select.transform(xTrain)
    xTest_trans = select.transform(xTest)
    headers_trans = select.get_feature_names_out(headers)
    
    mask = select.get_support()
    plt.matshow(mask.reshape(1,-1), cmap='gray_r')
    plt.xlabel("Select based on Linear Regression")
    plt.yticks(())
    print("Remaining Headers =", headers_trans)
    print("coef:\n", select.estimator_.coef_)
    print("Features:\n", xTrain_trans.shape[1])
    return xTrain_trans, xTest_trans, yTrain, yTest, headers_trans

def usePCA(xTrain, xTest, headers):
    tenPercent = int(xTrain.shape[1]/10)
    print("Ten percent of current features", tenPercent)
    pca = PCA(n_components=tenPercent).fit(xTrain)
    xTrain_pca = pca.transform(xTrain)
    xTest_pca = pca.transform(xTest)
    headers_trans = pca.get_feature_names_out(headers)
    print("shapes for Garrett: ", xTrain_pca.shape)
    print("shapes for Garriett: ", xTest_pca.shape)
    return xTrain_pca, xTest_pca, headers_trans

def machineLearningModel(xTrain, xTest, yTrain, yTest, headers):
    #poggers
    print()
    xTrain_Temp = np.array(xTrain, dtype=np.float64)
    yTrain_Temp = np.array(yTrain, dtype=np.float64)
    xTest_Temp = np.array(xTest, dtype=np.float64)
    yTest_Temp = np.array(yTest, dtype=np.float64)
    
    parameters = {'epsilon': [20, 50, 100, 1000, 10000],
                  'C': [20, 50, 100, 1000, 10000]}
    

    all_scores = np.array([])
    for ep in parameters['epsilon']:
        scores = np.array([])
        for c in parameters['C']:
            svm = SVR(C=c, epsilon=ep)
            svm.fit(xTrain_Temp, yTrain_Temp)
            
            scores = np.append(scores, float("{:.2f}".format(r2_score(yTest_Temp, svm.predict(xTest_Temp)))))
            """
            print("Epsilon =", ep, ",C =", c, "Score =", svm.score(xTest_Temp, yTest_Temp))
            print("R2 =", float("{:.2f}".format(r2_score(yTest_Temp, svm.predict(xTest_Temp)))))
            print("RMSE: ", mean_absolute_error(yTest_Temp, svm.predict(xTest_Temp)))
            """
        all_scores = np.append(all_scores, scores)
            
    all_scores = np.reshape(all_scores, (-1,len(scores)))
    all_scores = np.transpose(all_scores)
    print(all_scores)
    print(all_scores.shape)
    x = np.arange(len(parameters['epsilon']))
    width = .15
    multiplier = 0
    
    fig, ax = plt.subplots(layout='constrained')
    
    for index, c in enumerate(parameters['C']):
        offset = width * multiplier
        print(all_scores[index], "\n")
        rects = ax.bar(x + offset, all_scores[index], width, label="Cost: " + str(c))
        multiplier += 1
    
    ax.set_ylabel('R Squared')
    ax.set_xlabel('Epsilon')
    ax.set_title('SVR Regression Scores by parameter')
    ax.set_xticks(x + width, parameters['epsilon'])
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 1)
    


processedFile = preprocessCSV(filePathReg)
data, targets, headers = csvToNumPyArray(processedFile)
xTrain, xTest, yTrain, yTest, headers = reduceFeatures(data, targets, headers)
xTrain_reduced, xTest_reduced, headers_reduced = usePCA(xTrain, xTest, headers)
machineLearningModel(xTrain_reduced, xTest_reduced, yTrain, yTest, headers_reduced)

