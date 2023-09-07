# -*- coding: utf-8 -*-
"""
regressionTask - Final Project
Written By: Cameron McCready
Date Of Completion: August 31, 2023

This file is used to write and execute code for the Final Project.in 
Machine Learning. The program was used to complete all the investigations 
brought on to analyze game data in val_stats_cut.csv
    
Investigation 1 - Game data determining Winrate (Regression)

Investigation 2 - Game data determining Winrate with feature removal (Regression)

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

# Model Development (Model Selection and Tuning For Best Parameters)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression 

# Math
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

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
                                               columns = ['win_percent'])], 
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
data['agent_1'].value_counts().plot(kind = 'bar')

# Set up Histogram to show kills data
# Use pd.DataFrame()
fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
n, bins, patches = ax1.hist(pd.DataFrame(data, 
                                         columns = ['kills']), 
                                         color="blue", 
                                         edgecolor='black', 
                                         linewidth=1.2)
ax1.set_title('kills Histogram')
ax1.set_xlabel('kills')
ax1.set_ylabel('Frequency')

# Set up Histogram to show deaths data
# Use pd.DataFrame()
fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)
n, bins, patches = ax2.hist(pd.DataFrame(data, 
                                         columns = ['deaths']), 
                                         color="blue", 
                                         edgecolor='black', 
                                         linewidth=1.2)
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

# Verify if there is missing values in the data
print("\nCount total NaN at each column in a DataFrame:\n")
print((data.isnull().sum()))

# (Our data does not have missing values. Hooray!)

#------------------------------------------------------------------
# Column Transformation (One Hot Encoding, Standrad Scalar, etc)
#------------------------------------------------------------------

print("\n One Hot Encoding and Standard Scaler\n")

#----------------------
# One-Hot Encoding
#----------------------

# Print Shape For Later Verification
print("Data Before One Hot Encoding:\n", np.shape(data))

# Remove target column again for One-Hot Encoding (New features will be added in
# after the target column and we want to maintain the format that was established)
data_features = data.drop(columns = ['win_percent'])

# One Hot Encode Categorical Data in Features
data_features =  pd.get_dummies(data_features)

# Readd target column
data =  pd.concat([data_features, pd.DataFrame(data, columns = ['win_percent'])], axis = 1)

# Print to Verify
print("\n One Hot Encoding\n")
print(data)
print("Data After One Hot Encoding:\n", np.shape(data))

#----------------------
# Standard Scaler
#----------------------

# Use Standard Scaler to Scale all the data to standardized form
scaler = StandardScaler()
data = pd.DataFrame(scaler.fit_transform(data), columns = data.columns)

# Print to Verify
print("\nStandard Scaler\n")
print(data)

#----------------------------------------------------------------------
# Remove Insignificant Features (Feature Removal Techniques)
#----------------------------------------------------------------------

# Use Model Based Feature Selection to see how feature removal would fair in
# the prediction of data. This is be commented out for Investigation 1

print("Model Based Feature Selection\n") 

print("data.shape BEFORE REDUCTION:\n", data.shape) 

data_features = data.drop(columns = ['win_percent'])

# Use linear regression as your base model
select = SelectFromModel(Ridge(), threshold = "median")

# Fit features and target
select.fit(data_features, data['win_percent'])

# Set data after SelectFromModel feature reduction
features_selected = pd.DataFrame(select.transform(data_features))

# Readd target column
data =  pd.concat([features_selected, pd.DataFrame(data, columns = ['win_percent'])], axis = 1)

# Verify reduction with shape
print("data.shape AFTER REDUCTION:\n", data.shape)

# Time to see what features were removed
mask = select.get_support()
plt.matshow(mask.reshape(1, -1), cmap = 'gray_r')
plt.xlabel("Sample Index")
plt.yticks(())

print(data)

# Add Space
print("\n")

#----------------------------------------------------------------------
# Linear, Non-Linear Model SVM, Tree Based Model, and NN Model
# Development With Parameter Tuning
#----------------------------------------------------------------------

# Readjust the data so that target is removed once more 
data_features = data.drop(columns = ['win_percent'])

# Test Split
DATA_TRAIN, DATA_TEST, TARGET_TRAIN, TARGET_TEST = train_test_split(
                                                   data_features, 
                                                   data['win_percent'],
                                                   test_size = 0.8, 
                                                   random_state = 42)

# Get Best Parameters For Parameter Tuning


# Evaluate for best parameters for SVM
param_grid1 = {'C': [0.1, 1, 10], 
               'gamma': [0.01, 0.001, 0.0001],
               'kernel': ['rbf']} 

svmRegressor = GridSearchCV(svm.SVR(),param_grid1,refit=True,verbose=2)
svmRegressor.fit(DATA_TRAIN, TARGET_TRAIN)

print("Best value for Non-Linear SVM:\n", svmRegressor.best_params_)

# Evalauate for best parameters for Tree Based Model
param_grid2 = {"max_depth" : [1,3,5,7,9,11,12,13]}

treeRegressor = GridSearchCV(DecisionTreeRegressor(),param_grid2, cv=3,verbose=3)
treeRegressor.fit(DATA_TRAIN, TARGET_TRAIN)

print("Best value for Tree Based Model:\n", treeRegressor.best_params_)

# Evaluate for best parameters for NN
param_grid3 = {
    'hidden_layer_sizes': [1, 5, 10, 20],
    'max_iter': [500]}

NNRegressor = GridSearchCV(MLPRegressor(), param_grid3, n_jobs= -1, cv=5)
NNRegressor.fit(DATA_TRAIN, TARGET_TRAIN)

print("Best value for NN:\n", NNRegressor.best_params_)


#----------------------------------------------------------------------
# Regression Metrics
#----------------------------------------------------------------------

# Create models and calculate metric values (RMSE and R2)
# We will do this for Linear, SVM, Tree-Based, and NN

# Linear
lr = LinearRegression()
lr.fit(DATA_TRAIN, TARGET_TRAIN)
pred_lr = lr.predict(DATA_TEST)
print("Linear Regression RMSE:\n", np.sqrt(mean_squared_error(TARGET_TEST, pred_lr)))
print("Linear Regression R^2:\n", r2_score(TARGET_TEST, pred_lr))

print("\n")

# SVM
SVM = svm.SVR(C=10, gamma=0.001, kernel = 'rbf') 
SVM.fit(DATA_TRAIN, TARGET_TRAIN)
pred_svm = SVM.predict(DATA_TEST)
rmse_svm = np.sqrt(mean_squared_error(TARGET_TEST, pred_svm))
r2_svm = r2_score(TARGET_TEST, pred_svm)
print("Non-Linear SVM Regression RMSE:", rmse_svm)
print("Non-Linear SVM Regression R^2:", r2_svm)

print("\n")

# Tree Based
TREE = DecisionTreeRegressor(max_depth= 3)
TREE.fit(DATA_TRAIN, TARGET_TRAIN)
pred_tree = TREE.predict(DATA_TEST)
print("Tree-Based Regression RMSE:", np.sqrt(mean_squared_error(TARGET_TEST, pred_tree)))
print("Tree-Based Regression R^2:", r2_score(TARGET_TEST, pred_tree))

print("\n")

# NN
NN = MLPRegressor(max_iter=500, hidden_layer_sizes = 20)
NN.fit(DATA_TRAIN, TARGET_TRAIN)
pred_nn = NN.predict(DATA_TEST)
rmse_nn = np.sqrt(mean_squared_error(TARGET_TEST, pred_nn))
r2_nn =  r2_score(TARGET_TEST, pred_nn)
print("NN Regression RMSE:", rmse_nn)
print("NN Regression R^2:", r2_nn)



