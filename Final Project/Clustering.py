# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 22:22:31 2023

@author: gholt
"""
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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_curve
from sklearn.cluster import KMeans


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# Global Variables

def main():
    print("Processing csv")
    processedData = processCSV()
    
    print("\nDetermining num of clusters")
    determineClusterNum(processedData)
    
    print("\nRunning kmeans with no target")
    dataC0, dataC1, dataC2, dataC3, kmeans = runKMeans(processedData)
    
    print("\nCreating scatters")
    createScatters(dataC0, dataC1, dataC2, dataC3, kmeans)
    
    print("\nRunning kmeans with target")
    dataStatsWithTarget(processedData)
    return 0

def processCSV():
    data = pd.read_csv("val_stats_cut.csv")
    print(data.columns, "\n")
    
    np.set_printoptions(suppress=True)
    
    data_dummies = pd.get_dummies(data)
    print(data_dummies.columns)
    
    return data_dummies

def determineClusterNum(data):
    wcss = []
    for i in range(1,10):
        kmeans = KMeans(n_clusters=i, 
                         random_state=42, 
                         n_init='auto')
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    #plot elbow curve
    plt.plot(np.arange(1,10), wcss)
    plt.xlabel('Clusters')
    plt.ylabel('SSE')
    plt.title("Elbow Curve w/ no Target")
    plt.show()
    
    kmeans = KMeans()
    parameters = {
        'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9 , 10],
        'random_state': [42],
        'n_init':['auto']
    }
    
    grid_search = GridSearchCV(kmeans, parameters, cv=5)
    
    grid_search.fit(data)
    
    print(grid_search.best_params_)
    
    return 0

def runKMeans(data):
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    kmeans.fit(data)

    plt.hist(kmeans.labels_, facecolor="blue", align='mid', orientation='horizontal', bins=5)
    plt.yticks([0,1,2])
    plt.xlabel("Number of Occurrences")
    plt.title("Clusters Based off ValStats")
    plt.show()

    clusters = kmeans.predict(data)

    #print(kmeans.cluster_centers_, "\n")
    
    """
    for num in range(len(kmeans.cluster_centers_[0])):
        print(data.columns[num])
        print(kmeans.cluster_centers_[0][num], 
              kmeans.cluster_centers_[1][num], 
              kmeans.cluster_centers_[2][num], 
              kmeans.cluster_centers_[3][num], '\n')
    """
    count = 0
    for num in range(len(kmeans.cluster_centers_[0])):
        num1 = kmeans.cluster_centers_[0][num]
        num2 = kmeans.cluster_centers_[1][num]
        num3 = kmeans.cluster_centers_[2][num]
        if max(num1, num2, num3) - min(num1, num2, num3) > 10:
            count+=1
            print(data.columns[num])
            print(num1, 
                  num2, 
                  num3, '\n')
    print("Num of features", count)
        
    data['cluster'] = clusters

    dataC0 = data[data.cluster==0]
    dataC1 = data[data.cluster==1]
    dataC2 = data[data.cluster==2]
    dataC3 = data
    
    """
    dataC0.to_csv('dataC0.csv', index=False)
    dataC1.to_csv('dataC1.csv', index=False)
    dataC2.to_csv('dataC2.csv', index=False)
    dataC3.to_csv('dataC3.csv', index=False)
    """
    
    return dataC0, dataC1, dataC2, dataC3, kmeans

def createScatters(dataC0, dataC1, dataC2, dataC3, kmeans):
    kplot = plt.axes(projection='3d')
    # Data for three-dimensional scattered points
    kplot.scatter3D(dataC0.deaths, 
                    dataC0.headshots, 
                    dataC0.kills, 
                    c='red', 
                    label='Cluster 1')
    kplot.scatter3D(dataC1.deaths, 
                    dataC1.headshots, 
                    dataC1.kills, 
                    c='green', 
                    label='Cluster 2')
    kplot.scatter3D(dataC2.deaths, 
                    dataC2.headshots, 
                    dataC2.kills, 
                    c='blue', 
                    label='Cluster 3')
    """
    kplot.scatter3D(dataC3.deaths, 
                    dataC3.headshots, 
                    dataC3.kills, 
                    c='orange', 
                    label='Cluster 4')
    
    plt.scatter(kmeans.cluster_centers_[:,0], 
                kmeans.cluster_centers_[:,1], 
                color='indigo', 
                s=200)
    """
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Deaths vs. Headshots vs Kills")
    plt.show()
    
    kplot = plt.axes(projection='3d')
    # Data for three-dimensional scattered points
    kplot.scatter3D(dataC0.headshots, 
                    dataC0.clutches, 
                    dataC0.flawless, 
                    c='red', 
                    label='Cluster 1')
    kplot.scatter3D(dataC1.headshots, 
                    dataC1.clutches, 
                    dataC1.flawless, 
                    c='green', 
                    label='Cluster 2')
    kplot.scatter3D(dataC2.headshots, 
                    dataC2.clutches, 
                    dataC2.flawless, 
                    c='blue', 
                    label='Cluster 3')
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Headshots vs Clutches vs Flawless")
    plt.show()
    
    kplot = plt.axes(projection='3d')
    # Data for three-dimensional scattered points
    kplot.scatter3D(dataC0.gun1_kills, 
                    dataC0.first_bloods, 
                    dataC0.assists, 
                    c='red', 
                    label='Cluster 1')
    kplot.scatter3D(dataC1.gun1_kills, 
                    dataC1.first_bloods, 
                    dataC1.assists, 
                    c='green', 
                    label='Cluster 2')
    kplot.scatter3D(dataC2.gun1_kills, 
                    dataC2.first_bloods, 
                    dataC2.assists, 
                    c='blue', 
                    label='Cluster 3')
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Gun1 Kills vs First Bloods vs Assists")
    plt.show()
    
    return 0

def dataStatsWithoutTarget(data):
    
    return 0

def dataStatsWithTarget(data):
    data_features = data.drop(columns = ['kd_ratio'])
    
    DATA_TRAIN, DATA_TEST, TARGET_TRAIN, TARGET_TEST = train_test_split(
                                                       data_features, 
                                                       data['kd_ratio'],
                                                       test_size = 0.7, 
                                                       random_state = 42)
    
    wcss = []
    for i in range(1,10):
        kmeans = KMeans(n_clusters=i,
                         random_state=42, 
                         n_init='auto')
        kmeans.fit(DATA_TRAIN, TARGET_TRAIN)
        wcss.append(kmeans.inertia_)
    #plot elbow curve
    plt.plot(np.arange(1,10), wcss)
    plt.xlabel('Clusters')
    plt.ylabel('SSE')
    plt.title("Elbow Curve w/ Target")
    plt.show()
    
    # Run kmeans on the train data set
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    kmeans.fit(DATA_TRAIN, TARGET_TRAIN)

    plt.hist(kmeans.labels_, 
             facecolor="blue", 
             align='mid', 
             orientation='horizontal', 
             bins=5)
    plt.yticks([0,1,2])
    plt.xlabel("Number of Occurrences")
    plt.title("Clusters Based off ValStats")
    plt.show()
    
    count = 0
    for num in range(len(kmeans.cluster_centers_[0])):
        num1 = kmeans.cluster_centers_[0][num]
        num2 = kmeans.cluster_centers_[1][num]
        num3 = kmeans.cluster_centers_[2][num]
        if max(num1, num2, num3) - min(num1, num2, num3) > 10:
            count+=1
            print(data.columns[num])
            print(num1, 
                  num2, 
                  num3, '\n')
    print("Num of features", count)
    
    print(kmeans.score(DATA_TEST, TARGET_TEST))
    
    print(r2_score(TARGET_TEST, kmeans.predict(DATA_TEST)))
    print(np.sqrt(mean_squared_error(TARGET_TEST, kmeans.predict(DATA_TEST))))
    
    return 0

main()