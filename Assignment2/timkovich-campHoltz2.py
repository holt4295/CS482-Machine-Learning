# -*- coding: utf-8 -*-
"""
Garrett Holtz
Sean Timkovich-Camp

Assignment 2

TO DO:
    1. Print first 5 datasets
    2. Scatterplot
    3. Split training and test data
    4. Compute best # of neighbors to use
    5. Train data using 5-fold StratifiedKFold
"""
import matplotlib.pyplot as plt
import numpy as np
import csv


filePath = "modified_wdbc.csv"
headers = []
target_name = []
targets = np.array([])
data = []

"""

"""
def csvToNumPyArray():
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
    
    print("data-numpy array of shape ", data.shape)
    data = np.delete(data, 0, 1)
    
    print("target-numpy array of shape ", len(targets))
    
    target_name = headers.pop(0)
    headers.pop() #Remove trailing header
    
    print("Target Name: ", target_name)
    print("Feature Names: ", headers)

"""

"""
def createHistograms():
    radius1Data = np.array([])
    texture1Data = np.array([])
    for i in range(len(data)):
        radius1Data = np.append(radius1Data, data[i, 0])
        texture1Data = np.append(texture1Data, data[i, 1])
    #print(radius1Data)
    
    #Radius1 Hist
    fig1, ax1 = plt.subplots()
    plt.hist(radius1Data, facecolor="blue", edgecolor="gray")

    #ax1.set_xticks(bins)
    fig1.suptitle("Radius1 Data Histogram")
    ax1.set_xlabel("Radius1 Data Value")
    ax1.set_ylabel("Frequency")
    
    #Texture1 Hist
    fig2, ax2 = plt.subplots()
    plt.hist(texture1Data, facecolor="blue", edgecolor="gray")

    #ax2.set_xticks(bins)
    fig2.suptitle("Texture1 Data Histogram")
    ax2.set_xlabel("Texture1 Data Value")
    ax2.set_ylabel("Frequency")

"""

"""
def createScatterPlot():
    #perimeter x symmetry
    
    fig, ax = plt.subplots()
    #plt.scatter(perimeterData, symmetryData, c=targets, label=targets)
    
    color1 = 'tab:blue'
    color2 = "tab:orange"
    
    for i in range(len(data)):
        if targets[i] == 0:
            ax.scatter(data[i, 2], data[i, 27], c=color1, label=color1, alpha=0.3)
        else:
            ax.scatter(data[i, 2], data[i, 27], c=color2, label=color2, alpha=0.3)
    
    fig.suptitle("Perimeter vs. Symmetry Scatter Plot")
    ax.set_xlabel("Perimeter")
    ax.set_ylabel("Symmetry")
    ax.legend() # Add a legend.
    #ax.grid(True)
    
    plt.show()


csvToNumPyArray()
#createHistograms()
createScatterPlot()

