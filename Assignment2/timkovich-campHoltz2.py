# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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
                    targets = np.append(targets, [0])
                else:
                    targets = np.append(targets, [1])
                    
    data = np.genfromtxt(filePath, delimiter=",", skip_header=1, dtype=float)
    print("data-numpy array of shape ", data.shape)
    print("target-numpy array of shape ", len(targets))
    target_name = headers.pop(0)
    headers.pop() #Remove trailing header
    print("Target Name: ", target_name)
    print("Feature Names: ", headers)
    data = np.delete(data, 0, 1)

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
    return 0


csvToNumPyArray()
#createHistograms()


