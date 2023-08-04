# -*- coding: utf-8 -*-
"""
Garrett Holtz
Sean Timkovich-Camp

Assignment 2
"""

"""
Imports
"""
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
import csv
import math

"""
Global Declarations
"""
filePath = "modified_wdbc.csv"
headers = []
target_name = []
targets = np.array([])
data = []

X_train = np.array([])
X_test = np.array([])
Y_train = np.array([])
Y_test = np.array([])

"""
Import data from .csv to multiple numpy arrays
"""
def csvToNumPyArray():
    print("\nRunning csvToNumPyArray")
    global data # To modify global variable, needs to be redefined here
    global targets
    global headers
    with open(filePath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row_num, row in enumerate(csv_reader):
            if row_num == 0:
                headers = row
            else:
                if row[0] == "M":
                    targets = np.append(targets, 0)
                else:
                    targets = np.append(targets, 1)
                    
    data = np.genfromtxt(filePath, delimiter=",", skip_header=1, dtype=float)
    np.set_printoptions(precision=3, suppress=True)
    
    print("\ndata-numpy array of shape ", data.shape)
    data = np.delete(data, 0, 1)
    
    print("target-numpy array of shape ", len(targets))
    
    target_name = headers.pop(0)
    headers.pop() #Remove trailing header
    
    print("Target Name: ", target_name)
    print("Feature Names: ", headers)
    
    for index, row in enumerate(data):
        if index > 4:
            break
        print("Row ", index, ": ", row)

"""
Create Histograms based off features 1 and 2
"""
def createHistograms():
    print("\nRunning createHistograms")
    radius1Data = np.array([])
    texture1Data = np.array([])
    for i in range(len(data)):
        radius1Data = np.append(radius1Data, data[i, 0])
        texture1Data = np.append(texture1Data, data[i, 1])
    #print(radius1Data)
    
    #Radius1 Hist
    fig1, ax1 = plt.subplots()

    fig1.suptitle("Radius1 Data Histogram")
    ax1.set_xlabel("Radius1 Data Value")
    ax1.set_ylabel("Frequency")
    ax1.hist(radius1Data, facecolor="blue", edgecolor="gray")
    
    #Texture1 Hist
    fig2, ax2 = plt.subplots()
    
    fig2.suptitle("Texture1 Data Histogram")
    ax2.set_xlabel("Texture1 Data Value")
    ax2.set_ylabel("Frequency")
    ax2.hist(texture1Data, facecolor="blue", edgecolor="gray")

"""
Create a scatterplot comparing features 3 and 28
"""
def createScatterPlot():
    print("\nRunning createScatterPlot")
    #perimeter x symmetry
    
    fig, ax = plt.subplots()
    
    color1 = 'tab:red'
    color2 = "tab:blue"
    data0_perimeter = np.array([])
    data0_symmetry = np.array([])
    data1_perimeter = np.array([])
    data1_symmetry = np.array([])
    for index, row in enumerate(data):
        if targets[index] == 0: 
            data0_perimeter = np.append(data0_perimeter, row[2])
            data0_symmetry = np.append(data0_symmetry, row[27])
        else:
            data1_perimeter = np.append(data1_perimeter, row[2])
            data1_symmetry = np.append(data1_symmetry, row[27])
        
    
    ax.scatter(data0_perimeter, data0_symmetry, c=color1, label="Malignant", alpha=0.5)
    ax.scatter(data1_perimeter, data1_symmetry, c=color2, label="Benign", alpha=0.5)
    
    fig.suptitle("Perimeter vs. Symmetry Scatter Plot")
    ax.set_xlabel("Perimeter")
    ax.set_ylabel("Symmetry")
    ax.legend() # Add a legend.
    #ax.grid(True)
    
    plt.show()

"""
Find the best number of neighbors to be used for K-NN
"""
def findK():
    print("\nRunning findK")
    global X_train
    global X_test
    global Y_train
    global Y_test
    X_train, X_test, Y_train, Y_test = train_test_split(data, targets, random_state=42)
    
    print("\n", len(X_train), len(Y_train), len(X_test), len(Y_test))
    
    trainingAccuracy = np.array([])
    testingAccuracy = np.array([])
    
    numSamples = int(math.sqrt(len(data)) + 3)
    neighborsRange = range(1, numSamples, 2) #Only want odd numbers so iterate by 2
    
    for n_neighbors in neighborsRange:
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        clf.fit(X_train, Y_train)
        trainingAccuracy = np.append(trainingAccuracy, clf.score(X_train, Y_train))
        testingAccuracy = np.append(testingAccuracy, clf.score(X_test, Y_test))

    fig, ax = plt.subplots()
    fig.suptitle("Accuracy vs. N_Neighbors")
    ax.plot(neighborsRange, trainingAccuracy, label="Training accuracy")
    ax.plot(neighborsRange, testingAccuracy, label="Test accuracy")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("n_neighbors")
    ax.set_xticks(neighborsRange)
    ax.legend()

"""
Cross validate with StratifiedKFold the number of neighbors found against 
the training and test data
"""
def crossValidation():
    print("\nRunning crossValidation with StratifiedKFold")
    clf = KNeighborsClassifier(n_neighbors=5)
    scores = cross_val_score(clf, X_train, Y_train, cv=5)
    print("\nCross-validation with StratifiedKFold training scores:\n {}".format(scores))
    
    scores = cross_val_score(clf, X_test, Y_test, cv=5)
    print("\nCross-validation with StratifiedKFold testing scores:\n {}".format(scores))

"""
Run functions
"""
csvToNumPyArray()
createHistograms()
createScatterPlot()
findK()
crossValidation()

