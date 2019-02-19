# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 19:44:13 2018

@author: qiang
"""


import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt

# import the dataset
dataset = pd.read_csv(r'C:\Users\qiang\Desktop\2018 fall\5825\homework\5\wtkg.csv')

# get 10 samples from dataset
train_data = dataset.sample(n=10)
# create x
train_data_X = train_data.loc[:,'WTKG']
# create y
train_data_Y = train_data.loc[:,'Overweight']

# convert train_set and test_set to array
train_data_Xarray = train_data_X.values.ravel()
train_data_Yarray = train_data_Y.values.ravel()

# show train data
plt.scatter(train_data_Xarray,train_data_Yarray)
plt.show()


class ThreshClassifier():
    """
    train a threshold classifer
    predict according the classifier
    
    """
    
    def __init__(self):
        self.v = 0
        self.direction = 0
    
    # train the calssifer
    def train(self, x, y, w):
        # intialize the error and minerror
        error = 0
        minerror = 1    
        for v in np.arange(4000,15000,1000):
            # caculate the number of misclassify
            for direction in [0,1]:
                if direction == 0:
                    error = (((x < v) - 0.5)*2 != y)
                else:
                    error = (((x > v) - 0.5)*2 != y)
                # canculate the errow  
                error = sum(error * w)
                if error < minerror:
                    minerror = error
                    self.v = v
                    self.direction = direction
        return minerror
    
    # predict according the classifier
    def predict(self, x):
        if self.direction == 0:
            return ((x < self.v) - 0.5)*2
        else:
            return ((x > self.v) - 0.5)*2
    
    
class AdaBoost():
    """
    base classifier is the ThreshClassifier
    train new classifier according AdaBoost algorihm
    """
    def __init__(self, classifier = ThreshClassifier):
        self.classifier = classifier  # base classifier
        self.classifiers = []     # final classifer
        self.alphas = []          # alpha list
    def train(self, x, y):        
        # the number of data samples
        n = x.shape[0]
        # iteration time
        M = 2
        # initial weights
        w_m = np.array([1 / n] * n)
        print("The initial weights are:")
        print(w_m)

        for m in range(M):
            classifier_m = self.classifier()
            e_m = classifier_m.train(x, y, w_m)
            print("The error in this iteration is: %f" %(e_m))
            
            # calculate the alpha 
            alpha_m = 1 / 2 * np.log((1-e_m)/e_m)
            print("The alpha in this iteration is: %f" %(alpha_m))
            
            # calculate the new weight
            w_m = w_m * np.exp(-alpha_m*y*classifier_m.predict(x))
            
            # calculate the normalization 
            z_m = np.sum(w_m)
            
            #calculate the next iteration weightss
            w_m = w_m / z_m
            print("The new weights in the next iteration are:")
            print(w_m)
            
            self.classifiers.append(classifier_m)
            self.alphas.append(alpha_m)
            
    def predict(self, x):
        n = x.shape[0]
        results = np.zeros(n)
        for alpha, classifier in zip(self.alphas, self.classifiers):
            results += alpha * classifier.predict(x)
        return ((results > 0) - 0.5) * 2

ab = AdaBoost()
ab.train(train_data_Xarray, train_data_Yarray)
ab.predict(train_data_Xarray)
tc = ThreshClassifier()
tc.predict(train_data_Xarray)
train_data_Yarray
