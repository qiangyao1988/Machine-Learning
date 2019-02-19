#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2018/11/4 14:47
# @Author  : Yao Qiang
# @Author E-mail: gp8020@wayne.edu
# @File    : k-means.py
# @Version    : v1.0
# @Software: Spyder & Jupyter Notebook
# @Function: k-means clustering


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance_matrix
from sklearn.metrics.cluster import adjusted_rand_score
from scipy.misc import comb
from itertools import combinations


def PCA(dataset):
    '''Do PCA processing on dataset

    Use the PCA fuction from homework3
    Return the dataset after PCA processing

    Args:
        dataset: the dataset to deal with

    Return:
        dataset_transformed:the dataset after PCA processing
    '''

    # standard the dataset
    dataset_std = StandardScaler().fit_transform(dataset)

    # transfer the matrix of dataset
    dataFeatures = dataset_std.T

    # calculate the covariance
    covariance_matrix = np.cov(dataFeatures)

    # caculate the eigenvaluse and eigenvetors
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

    # sort the eigenvalues
    index = eigen_values.argsort()[::-1]
    eigen_values_sorted = eigen_values[index]

    # get the i largest eigenvalues
    sum_sort = eigen_values_sorted.sum()
    sum = 0
    for i in np.arange(len(eigen_values_sorted)):
        if sum / sum_sort < 0.9:
            sum = sum + eigen_values_sorted[i]
        else:
            break

    # select the i largest eigenvalues and eigenvectors
    dataset_transformed_column_size = i
    eigen_vectors_select = eigen_vectors[:,
                                         index][:, :dataset_transformed_column_size]

    # project the data on the new  bases
    dataset_transformed = dataset.dot(eigen_vectors_select)

    return dataset_transformed


def InitRandomCenter(dataset, k):
    '''Create the InitialRandomCenter according to dataset and k

    Get centroides randomly according the range of each column of dataset
    Return the centroids

    Args:
        dataset: the dataset to deal with
        k: the number of clusters

    Return:
        centroids:the initial random centers of the k clusters
    '''
    n = dataset.shape[1]
    centroids = np.mat(np.zeros((k, n)))

    # transfer the dataset to get the initial random dataset
    for j in range(n):
        minj = min(dataset[:, j])
        maxj = max(dataset[:, j])
        # make sure the centroids are in the dataset
        rangej = float(maxj - minj)
        centroids[:, j] = np.mat(minj + rangej * np.random.rand(k, 1))
    return centroids


def k_means(dataset, k):
    '''Do k-means proccessing on dataset

    Process k-means in 3 steps

    Args:
        dataset: the dataset to deal with
        k: the number of clusters

    Return:
        centroids:the final centers of the k clusters
        ClusterAssment: the clusterassignment of the samples in dataset
    '''

    # The total number of samples
    m = dataset.shape[0]
    # n = dataset.shape[1]

    ClusterAssment = np.mat(np.zeros((m, 2)))

    # step1:
    # The cluster centers are initialized by randomly generated sample points
    centroids = InitRandomCenter(dataset, k)

    # Flag bit, if the sample classification changes before and after iteration,
    # the value is Tree; otherwise, it is False
    ClusterChanged = True

    # The number of iterations
    IterTime = 0

    # All sample allocation results are no longer changed then the iteration is terminated
    while ClusterChanged:
        ClusterChanged = False
        # step2:Assign the samples to the nearest cluster center
        for i in range(m):
            # The initial definition of the distance is infinite
            minDist = np.inf
            # Initialize the index value
            minIndex = -1

            # Calculate the distance between each sample and k center points
            for j in range(k):
                distJI = distance_matrix(centroids[j, :], dataset[i, :], p=2)
                # Determine if the distance is minimal
                if distJI < minDist:
                    # Update to the minimum distance
                    minDist = distJI
                    # Gets the corresponding cluster index
                    minIndex = j
            # If the last sample assignment is different from this one,
            # then set the flag clusterChanged True.
            if ClusterAssment[i, 0] != minIndex:
                ClusterChanged = True
            # Assign the sample to the nearest cluster
            ClusterAssment[i, :] = minIndex
        IterTime += 1
        # step3:Update the clustering center
        # After the sample distribution, the cluster center is recalculated
        for center in range(k):
            # Obtain all sample points of the cluster
            DataInCluster = dataset[np.nonzero(
                ClusterAssment[:, 0].A == center)[0]]
            # Update the clustering center：
            # axis=0 to calculate the mean along the column direction.
            centroids[center, :] = np.mean(DataInCluster, axis=0)
    return centroids, ClusterAssment


def getXY(dataset):
    '''Select X,Y dimensions of dataset

    Iterate through the dataset and extract the first two columns
    as the x and y coordinates of the data

    Args:
        dataset: the dataset to deal with

    Return:
        X: the first colunmn of dataset
        Y: the second column of dataset
    '''
    m = dataset.shape[0]  # number of rows of dataset
    X = []
    Y = []
    for i in range(m):
        X.append(dataset[i, 0])
        Y.append(dataset[i, 1])
    return np.array(X), np.array(Y)


def showCluster(dataset, k, clusterAssment, centroids):
    '''Data Visualization

    Show dataset after k-means clustering accoridng the clusterAssment and centroids

    Args:
        dataset: the dataset to deal with
        k: the number of clusters
        clusterAssment: generated by k-means
        centroids:generated by k-means

    Return:
        X: the first colunmn of dataset
        Y: the second column of dataset
    '''
    fig = plt.figure()
    plt.title("K-means Clusterings")
    ax = fig.add_subplot(111)
    data = []

    # Extract data for each cluster
    for cent in range(k):
        # Gets the data belonging to the current cluster
        DataInCluster = dataset[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
        data.append(DataInCluster)
    # Draw a scatter plot of dataset points
    colors = ['r', 'g', 'b', 'y', 'c', 'm',
              'k', '#0000CD', '#90EE90', '#FF1493']
    markers = ['^', 'o', '*', 's', '.', '+', 'x', 'p', '<', '>']
    for cent, color, marker in zip(range(k), colors, markers):
        # print(data[cent])
        # print(data[cent].shape)
        X, Y = getXY(data[cent])
        ax.scatter(X, Y, s=80, c=color, marker=marker)

    ax.text(x=-9, y=0, s=k, withdash=False)

    # Draw a scatter plot of centroids
    centroidsX, centroidsY = getXY(centroids)
    ax.scatter(centroidsX, centroidsY, s=500,
               c='#8B0000', marker='o', alpha=1)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    plt.show()


def rand_index(labels_true, labels_pred):
    """given the true and predicted labels, 
    it will return the Rand Index.
    Args:
        labels_true: the real class labels 
        labels_pred:the predicted class labels
    Return:
        RAND_index
    """
    # create list of all combinations with the length of labels.
    my_pair = list(combinations(range(len(labels_true)), 2))

    def is_equal(x):
        return (x[0] == x[1])
    my_a = 0
    my_b = 0
    for i in range(len(my_pair)):
        if(is_equal((labels_true[my_pair[i][0]], labels_true[my_pair[i][1]])) == is_equal((labels_pred[my_pair[i][0]], labels_pred[my_pair[i][1]]))
           and is_equal((labels_pred[my_pair[i][0]], labels_pred[my_pair[i][1]])) == True):
            my_a += 1
        if(is_equal((labels_true[my_pair[i][0]], labels_true[my_pair[i][1]])) == is_equal((labels_pred[my_pair[i][0]], labels_pred[my_pair[i][1]]))
           and is_equal((labels_pred[my_pair[i][0]], labels_pred[my_pair[i][1]])) == False):
            my_b += 1
    my_denom = comb(len(labels_true), 2)
    RAND_index = (my_a + my_b) / my_denom
    return RAND_index


if __name__ == '__main__':

    # read data from csv
    mouse = pd.read_csv(
        r'C:\Users\qiang\Desktop\2018 fall\5825\homework\4\Data_Cortex_Nuclear.csv')

    # get the data without class, Genotype, Treatment and Behavior
    mouseX = mouse.iloc[:, 1:78]

    # fill the NaN data with mean
    mouseX = mouseX.fillna(mouseX.mean())

    # PCA the original data
    mouseX_PCA = PCA(mouseX)

    # transfer data into numpy.matrix
    mouseDataset = np.mat(mouseX_PCA)

    # the label of the class of the mice
    label_true = np.array(mouse['class']).T

    # the label of the class of the mice
    mice_class = pd.read_csv(
        r'C:\Users\qiang\Desktop\2018 fall\5825\homework\4\mice_class.csv')
    label_t = np.array(mice_class).T
    label_true = label_t.ravel()

    # k is the number of user-defined clusters
    for k in range(2, 11):
        mycentroids, myclusterAssment = k_means(mouseDataset, k)
        # data visualization after k-means clustering
        showCluster(mouseDataset, k, myclusterAssment, mycentroids)
        label_p = np.array(myclusterAssment[:, 1].T)
        label_pre = label_p.ravel()
        # calculate the RAND_index
        RAND_index = rand_index(label_true, label_pre)
        print("The RAND index of the %d-means clustering is %f" %
              (k, RAND_index))
        # calculate the Adjusted_RAND_index
        Adjusted_RAND_index = adjusted_rand_score(label_true, label_pre)
        print("The Adjusted RAND index of the %d-means clustering is %f" %
              (k, Adjusted_RAND_index))
