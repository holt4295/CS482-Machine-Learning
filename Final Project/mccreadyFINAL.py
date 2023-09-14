# -*- coding: utf-8 -*-
"""
mccreadyFINAL - Final Project
Written By: Cameron McCready
Date Of Completion: August 28, 2023

This file is used to write and execute code for the Final Project.in 
Machine Learning. The program was used to complete all the investigations 
brought on to analyze agent-oriented data in val_stats_cut.csv
    
Investigation 1 - Agents ONLY determining KD, Kad, and Winrate

Investigation 2 - Agents ONLY determining KD, Kad, and Winrate 
                  WITH feature reduction

Investigation 3 - Agents with other data determining KD, Kad, and Winrate

Investigation 4   Agents with other data determining KD, Kad, and Winrate 
                  WITH feature reduction

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

# Feature Reduction Oriented 
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA

# Model Development
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

# Math

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

#-----------------------------------------------------------
# Load and Manually Modify Data
#-----------------------------------------------------------

# The investigations use pd.read_csv to get information from val_stats_cut.csv

# Use pd.read
data = pd.read_csv("val_stats_cut.csv")

# Readjust the data so that target columns appear last in the table
# Use data.drop and pd.concat to do this
data_features = data.drop(columns = ['kd_ratio', 'kad_ratio', 'win_percent'])

data =  pd.concat([data_features, pd.DataFrame(data, columns =
                                               ['kd_ratio',
                                                'kad_ratio',
                                                'win_percent']
                                                 )], axis = 1)


# Remove gun 1 data columns. We won't be using these during investigations

data = data.drop(columns = ['gun1_name', 'gun1_head', 'gun1_body',
                            'gun1_legs', 'gun1_kills'])


# Here, we remove other features if we think they will be correlating 
# too well with predictions

# data = data.drop(columns = ['kills, 'deaths', 'assists'])

# Here, we can also remove two target columns for the sake of testing one target.
# 1. Target = kd_ratio
# 2. Target = kad_ratio
# 3. Target = win_percent

data = data.drop(columns = ['kad_ratio', 'win_percent'])
# data_adjusted = data.drop(columns = ['kd_ratio', 'win_percent'])
# data_adjusted = data.drop(columns = ['kad_ratio', 'kd_ratio'])

# Verify data using print()
print("\nRaw Data")
print(data)

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
data['agent_1'].value_counts().plot(kind = 'bar')
# Use value_counts() to plot rating
data['rating'].value_counts().plot(kind = 'bar')

# Set up Histogram to show all kills data
# Use pd.DataFrame() to plot kills Data
fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
n, bins, patches = ax1.hist(pd.DataFrame(data, columns = ['kills']), color="blue", edgecolor='black', linewidth=1.2)
ax1.set_title('kills Histogram')
ax1.set_xlabel('kills')
ax1.set_ylabel('Frequency')

# Set up Histogram to show all deaths data
# Use pd.DataFrame() to plot deaths Data
fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)
n, bins, patches = ax2.hist(pd.DataFrame(data, columns = ['deaths']), color="blue", edgecolor='black', linewidth=1.2)
ax2.set_title('deaths Histogram')
ax2.set_xlabel('deaths')
ax2.set_ylabel('Frequency')
   
# g) Pair Plot Target 

# Set up Pair Plot Target to show all kills and deaths data
# Use pd.DataFrame to plot both kills and deaths Data 
fig3 = plt.figure()
ax3 = fig3.add_subplot(1, 1, 1)

ax3.scatter(pd.DataFrame(data, columns = ['kills']), 
            pd.DataFrame(data, columns = ['deaths']), 
            color='red')

ax3.set_title('deaths v. kills Scatterplot')
ax3.set_xlabel('kills')
ax3.set_ylabel('deaths')

#------------------------------------------------------------------
# Fill Any Missing Values
#------------------------------------------------------------------

# (Our data does not have missing values. Hooray!)

#------------------------------------------------------------------
# Column Transformation (One Hot Encoding, Standrad Scalar, etc)
#------------------------------------------------------------------

print("\n One Hot Encoding and Standard Scaler\n")

# Remove target column
data_features = data.drop(columns = ['kd_ratio'])

# Print Shape For Later Verification
print(np.shape(data))

# One Hot Encode Categorical Data in Features
data_features =  pd.get_dummies(data_features)

# Readd target column
data =  pd.concat([data_features, pd.DataFrame(data, columns = ['kd_ratio'])], axis = 1)

# Print to Verify
print("\n One Hot Encoding\n")
print(data)
print(np.shape(data))

# Use Standard Scaler to Scale all the data
scaler = StandardScaler()
data = pd.DataFrame(scaler.fit_transform(data), columns = data.columns)

#Print to Verify
print("\n Standard Scaler\n")
print(data)


#----------------------------------------------------------------------
# Remove Insignificant Features (Feature Removal Techniques)
#----------------------------------------------------------------------

# We won't be removing data yet (for investigations 2 and 4)

# Model Based Feature Selection?
# PCA?

#----------------------------------------------------------------------
# Non-Linear Model SVM, Tree Based Model, NN, etc. 
# With Parameter Tuning For Best Performance
#----------------------------------------------------------------------

# Readjust the data so that target is removed 
data_features = data.drop(columns = ['kd_ratio'])

# Test Split
DATA_TRAIN, DATA_TEST, TARGET_TRAIN, TARGET_TEST = train_test_split(
                                                   data_features, 
                                                   data['kd_ratio'],
                                                   test_size = 0.8, 
                                                   random_state = 42)

# FIND BEST PARAMETERS

# Evalauate for best parameters for Tree Based Model
param_grid1 = {"max_depth" : [1,3,5,7,9,11,12,13],
               "random_state": [42]}

treeRegressor = GridSearchCV(DecisionTreeRegressor(),param_grid1, cv=3,verbose=3)
treeRegressor.fit(DATA_TRAIN, TARGET_TRAIN)

print("Best value for Tree Based Model:\n", treeRegressor.best_params_)


# Evaluate for best parameters for SVM
param_grid2 = {'C': [1, 5], 
               'gamma': [0.01, 0.001]} 

svmRegressor = GridSearchCV(svm.SVR(),param_grid2,refit=True,verbose=2)
svmRegressor.fit(DATA_TRAIN, TARGET_TRAIN)

print("Best value for Non-Linear SVM:\n", svmRegressor.best_params_)


# Evaluate for best parameters for NN
param_grid3 = {
    'hidden_layer_sizes': [1, 5, 10, 20],
    'max_iter': [500],
    "random_state": [42]}

NNRegressor = GridSearchCV(MLPRegressor(), param_grid3, n_jobs= -1, cv=5)
NNRegressor.fit(DATA_TRAIN, TARGET_TRAIN)

print("Best value for NN:\n", NNRegressor.best_params_)

# TEST MODEL (Metrics)

# Linear Regression
lr = LinearRegression()
lr.fit(DATA_TRAIN, TARGET_TRAIN)
pred_lr = lr.predict(DATA_TEST)
print("Linear Regression RMSE:\n", np.sqrt(mean_squared_error(TARGET_TEST, pred_lr)))
print("Linear Regression R^2:\n", r2_score(TARGET_TEST, pred_lr))

print("\n")

# Tree Based Model Regression
TREE = DecisionTreeRegressor(max_depth= 7, random_state=42)
TREE.fit(DATA_TRAIN, TARGET_TRAIN)
pred_tree = TREE.predict(DATA_TEST)
print("Tree-Based Regression RMSE:", np.sqrt(mean_squared_error(TARGET_TEST, pred_tree)))
print("Tree-Based Regression R^2:", r2_score(TARGET_TEST, pred_tree))

print("\n")

# Non-Linear Model SVM
NON_LINEAR = svm.SVR(random_state=42) 
NON_LINEAR.fit(DATA_TRAIN, TARGET_TRAIN)
pred_svm = NON_LINEAR.predict(DATA_TEST)
rmse_svm = np.sqrt(mean_squared_error(TARGET_TEST, pred_svm))
r2_svm = r2_score(TARGET_TEST, pred_svm)
print("Non-Linear SVM Regression RMSE:", rmse_svm)
print("Non-Linear SVM Regression R^2:", r2_svm)

print("\n")

# NN
NN = MLPRegressor(random_state=42, max_iter=500, hidden_layer_sizes = 20)
NN.fit(DATA_TRAIN, TARGET_TRAIN)
pred_nn = NN.predict(DATA_TEST)
rmse_nn = np.sqrt(mean_squared_error(TARGET_TEST, pred_nn))
r2_nn =  r2_score(TARGET_TEST, pred_nn)
print("NN Regression RMSE:", rmse_nn)
print("NN Regression R^2:", r2_nn)










