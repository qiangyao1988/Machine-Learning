# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

#read data from csv
mouse = pd.read_csv(r'C:\Users\qiang\Data_Cortex_Nuclear.csv')

#get the data without class, Genotype, Treatment and Behavior
mouseX = mouse.iloc[:,1:78]

#get the class of the dataset
mouseY = mouse.iloc[:,-1]

#get the Behavior of the dataset
mouse_Behavior = mouse.iloc[:,-2]

#get the Treatment of the dataset
mouse_Treatment = mouse.iloc[:,-3]

#get the Genotype of the dataset
mouse_Genotype = mouse.iloc[:,-4]

#fill the NaN data with mean
mouseX = mouseX.fillna(mouseX.mean())

#standard the dataset
mouseX_std = StandardScaler().fit_transform(mouseX)

#transfer the matrix of dataset 
mouseFeatures = mouseX_std.T 

#calculate the covariance
covariance_matrix = np.cov(mouseFeatures)

#caculate the eigenvaluse and eigenvetors
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

#sort the eigenvalues
index = eigen_values.argsort()[::-1]

eigen_values_sorted=eigen_values[index]



#plot the hitogram of the sorted eigenvalues
plt.bar(range(1,78),eigen_values_sorted,alpha=0.5,align='center',label='eigen_values')
plt.ylabel('eigen_values')
plt.xlabel('principle components')
plt.legend()
plt.show

#get the i largest eigenvalues
sum_sort=eigen_values_sorted.sum()
sum=0
for i in np.arange(len(eigen_values_sorted)):
    if sum/sum_sort<0.9:
        sum = sum+eigen_values_sorted[i]
    else:
        break   
print('We choose the largest {number} eigenvalues'.format(number=i))

#select the i largest eigenvalues and eigenvectors
mouseX_transformed_column_size = i
eigen_values_select= eigen_values[index][:mouseX_transformed_column_size]
eigen_vectors_select = eigen_vectors[:,index][:,:mouseX_transformed_column_size]

#project the data on the new  bases
mouseX_transformed =mouseX.dot(eigen_vectors_select)

#select the first two pca
X1=mouseX_transformed.iloc[:,0]
X2=mouseX_transformed.iloc[:,1]

#join the PC1 and PC2 with class
result_class = pd.DataFrame()
result_class['PC1'] = X1
result_class['PC2'] = X2
result_class['class'] = mouseY

#plaot the new dataset according the classes
sns.lmplot('PC1', 'PC2', data=result_class, fit_reg=False,  # x-axis, y-axis, data, no line
           scatter_kws={"s": 30}, # marker size
           hue="class") # color
# title
plt.title('PCA result')

#join the PC1 and PC2 with Genotype
result_Genotype = pd.DataFrame()
result_Genotype['PC1'] = X1
result_Genotype['PC2'] = X2
result_Genotype['Genotype'] = mouse_Genotype

#plaot the new dataset according the Genotype
sns.lmplot('PC1', 'PC2', data=result_Genotype, fit_reg=False,  # x-axis, y-axis, data, no line
           scatter_kws={"s": 30}, # marker size
           hue="Genotype") # color
# title
plt.title('PCA result')

#join the PC1 and PC2 with Treatment
result_Treatment = pd.DataFrame()
result_Treatment['PC1'] = X1
result_Treatment['PC2'] = X2
result_Treatment['Treatment'] = mouse_Treatment

#plaot the new dataset according the Treatment
sns.lmplot('PC1', 'PC2', data=result_Treatment, fit_reg=False,  # x-axis, y-axis, data, no line
           scatter_kws={"s": 30}, # marker size
           hue="Treatment") # color
# title
plt.title('PCA result')

#join the PC1 and PC2 with Behavior
result_Behavior = pd.DataFrame()
result_Behavior['PC1'] = X1
result_Behavior['PC2'] = X2
result_Behavior['Behavior'] = mouse_Behavior

#plaot the new dataset according the Treatment
sns.lmplot('PC1', 'PC2', data=result_Behavior, fit_reg=False,  # x-axis, y-axis, data, no line
           scatter_kws={"s": 30}, # marker size
           hue="Behavior") # color
# title
plt.title('PCA result')