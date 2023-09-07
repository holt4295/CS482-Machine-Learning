# -*- coding: utf-8 -*-
"""
classificationTask - Final Project
Written By: Cameron McCready
Date Of Completion: August 31, 2023

This file is used to write and execute code for the Final Project.in 
Machine Learning. The program was used to complete all the investigations 
brought on to analyze game data in val_stats_cut.csv
    
Investigation 1 - Game data determining Rating (Regression)

Investigation 2 - Game data determining Rating with feature removal (Regression)

The program's purpose is to evaluate a student's expertise with the
content taught in Machine Learning. This would be sufficent evidence
 that the student mastered lecture content.
"""
# Import Capabilities

# Basic Data Operatives
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing 

# Feature Reduction Oriented 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


# Model Development
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve

#-----------------------------------------------------------
# Load and Manually Modify Data
#-----------------------------------------------------------
# The investigations use pd.read_csv to get information from val_stats_cut.csv

# Use pd.read
data = pd.read_csv("val_stats_cut.csv")

# Print to verify
print(data)

# We are able to see the data in Pandas DataFrame form

# Readjust the data so that target column 'win_percent' is the last column
# We are also removing 'rating' as well since this is the target column for
# a classification task

# Use data.drop and pd.concat to do this
data_features = data.drop(columns = ['win_percent', 'rating'])

data =  pd.concat([data_features, pd.DataFrame(data, 
                                               columns = ['rating'])], 
                                               axis = 1)

# Print to verify
print(data)

# Data was successfully readjusted

#---------------------------------------------------------
# Meet the Data
#---------------------------------------------------------

# Print Data Section Information 
print("\n\n\nMeet The Data Section:\n")

# a) Number of features
print("Number Of Features:\n")
print(len(data.columns)-1)
 
# b) Names of the features
print("\nFeature Names:\n")

i1 = 0

for col in data.columns:
    if(i1 < len(data.columns)-1):
        print(col)
        i1 = i1 + 1
    else:
        # c) Name of target 
        print("\nName Of Target:\n")
        print(col)
 
# d) Number of samples
print("\nNumber Of Samples:\n")
print(len(data.index))

# e) First five rows of the data
print("\nFirst Five Rows Of The Data:\n")
print(data.iloc[:5])
 
# f) Histograms

# Set up bar plots to show all categorical data
# Use value_counts() to plot agent1
data['rating'].value_counts().plot(kind = 'bar')

#------------------------------------------------------------------
# Fill Any Missing Values
#------------------------------------------------------------------

# Verify if there is missing values in the data
print("\nCount total NaN at each column in a DataFrame:\n")
print((data.isnull().sum()))

# (Our data does not have missing values. Hooray!)

#---------------------------------------------------------------------------------
# Column Transformation (Label Encoding, One Hot Encoding, Standrad Scalar, etc)
#---------------------------------------------------------------------------------

#-----------------------------------------
# Label Encoding and One Hot Encoding
#-----------------------------------------

# Make a data features for later cat
data_features = data.drop(columns = ['rating'])

# One Hot Encode Categorical Data in Features
data_features =  pd.get_dummies(data_features)

# Setup label encoder for rating
le = preprocessing.LabelEncoder() 
le.fit(data['rating']) 
print(list(le.classes_))

# Replace categorical values with labels 
newTarget = le.transform(data['rating'])

# Readd the hot encoded features and label encoded target into data
data =  pd.concat([data_features, pd.DataFrame(newTarget, columns = ['rating'])], axis = 1)

# Print to Verify
print("\n Label Encoding and One-Hot Encoding\n")
print(data)

#----------------------
# Standard Scaler
#----------------------

# Drop target column again
data_features = data.drop(columns = ['rating'])

# Use Standard Scaler to Scale all the data to standardized form
scaler = StandardScaler()
data_features = pd.DataFrame(scaler.fit_transform(data_features), columns = data_features.columns)

# Readd the hot encoded features and label encoded target into data
data =  pd.concat([data_features, data['rating']], axis = 1)


# Print to Verify
print("\nStandard Scaler\n")
print(data)


# g) Correlation Map

correlationMap = data.corr()
print(correlationMap)
print("\n")

#----------------------------------------------------------------------
# Remove Insignificant Features (Feature Removal Techniques)
#----------------------------------------------------------------------

print("Data Reduction\n")

print(data.shape)

data_features =  data.drop(columns = ['rating'])

fs = SelectKBest(score_func=f_classif, k=27)

data_features = pd.DataFrame(fs.fit_transform(data_features, data['rating']))

# Readd the hot encoded features and label encoded target into data
data =  pd.concat([data_features, data['rating']], axis = 1)

print(data.shape)

print(data)

#----------------------------------------------------------------------
# Logistic Regression, Confusion Matrix Values, and ROC Curve
#----------------------------------------------------------------------

# Readjust the data so that target is removed once more 
data_features = data.drop(columns = ['rating'])

# Test Split
DATA_TRAIN, DATA_TEST, TARGET_TRAIN, TARGET_TEST = train_test_split(
                                                  data_features, 
                                                  data['rating'],
                                                  test_size = 0.8, 
                                                  random_state = 42)

logreg = LogisticRegression(max_iter = 10000).fit(DATA_TRAIN, TARGET_TRAIN)
pred_logreg = logreg.predict(DATA_TEST)
print("Logistic Regression Accuracy: {:.2f}\n".format(logreg.score(DATA_TEST, TARGET_TEST)))

# Metrics
confusion_matrix = confusion_matrix(TARGET_TEST, pred_logreg)
print("Logistic Regression Confusion Matrix Values + Calculations\n")
print("Confusion Matrix:")
print(confusion_matrix)

print("Stats For Immortal 1:\n")

TP = confusion_matrix[0][0]
FP = confusion_matrix[1][0] + confusion_matrix[2][0] + confusion_matrix[3][0]
FN = confusion_matrix[0][1] + confusion_matrix[0][2] + confusion_matrix[0][3]
TN = len(data.index) - TP - FP - FN

print("\n")
print("Precision:")
print(TP/(TP+FP))
print("Recall:")
print(TP/(TP + FN))
print("Specificity:")
print(TN/(TN + FP))
print("Accuracy:")
print((TP + TN) / (TP + TN + FP + FN))
print("\n")

print("Stats For Immortal 2:\n")

TP = confusion_matrix[1][1]
FP = confusion_matrix[0][1] + confusion_matrix[2][1] + confusion_matrix[3][1]
FN = confusion_matrix[1][0] + confusion_matrix[1][2] + confusion_matrix[1][3]
TN = len(data.index) - TP - FP - FN

print("\n")
print("Precision:")
print(TP/(TP+FP))
print("Recall:")
print(TP/(TP + FN))
print("Specificity:")
print(TN/(TN + FP))
print("Accuracy:")
print((TP + TN) / (TP + TN + FP + FN))
print("\n")

print("Stats For Immortal 3:\n")

TP = confusion_matrix[2][2]
FP = confusion_matrix[0][2] + confusion_matrix[1][2] + confusion_matrix[3][2]
FN = confusion_matrix[2][0] + confusion_matrix[2][1] + confusion_matrix[2][3]
TN = len(data.index) - TP - FP - FN

print("\n")
print("Precision:")
print(TP/(TP+FP))
print("Recall:")
print(TP/(TP + FN))
print("Specificity:")
print(TN/(TN + FP))
print("Accuracy:")
print((TP + TN) / (TP + TN + FP + FN))
print("\n")

print("Stats For Radiant:\n")

TP = confusion_matrix[3][3]
FP = confusion_matrix[0][3] + confusion_matrix[1][3] + confusion_matrix[2][3]
FN = confusion_matrix[3][0] + confusion_matrix[3][1] + confusion_matrix[3][2]
TN = len(data.index) - TP - FP - FN

print("\n")
print("Precision:")
print(TP/(TP+FP))
print("Recall:")
print(TP/(TP + FN))
print("Specificity:")
print(TN/(TN + FP))
print("Accuracy:")
print((TP + TN) / (TP + TN + FP + FN))
print("\n")






