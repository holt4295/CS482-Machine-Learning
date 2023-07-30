# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
import numpy as np
import csv

headers = []
target_name = []
targets = np.array([])
data = []

def csvToNumPyArray():
    filePath = "modified_wdbc.csv"
    with open(filePath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        target = np.array([])
        for row_num, row in enumerate(csv_reader):
            if row_num == 0:
                headers = row
            else:
                if row[0] == "M":
                    target = np.append(target, [0])
                else:
                    target = np.append(target, [1])
                
    targets = target
    #filePath = "wwdbc.csv"
    #inputArray = np.fromfile(filePath, count=-1, dtype = float, sep=",")
    data = np.genfromtxt(filePath, delimiter=",", skip_header=1, dtype=float)
    print("data-numpy array of shape ", data.shape)
    print("target-numpy array of shape ", len(target))
    target_name = headers.pop(0)
    headers.pop() #Remove trailing header
    print("Target Name: ", target_name)
    print("Feature Names: ", headers)
    data = np.delete(data, 0, 1)
    
csvToNumPyArray()


