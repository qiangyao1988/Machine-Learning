# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 17:03:48 2018

@author: qiang
"""

import numpy as np
import xport
import pandas as pd

# import the dataset
path = r"C:\Users\qiang\Desktop\2018 fall\5825\homework\project\LLCP2017.XPT"
with open(path,'rb') as f:
    df = xport.to_dataframe(f)
     
# select BMI5 from dataset
data_BMI = df['_BMI5']
#fill the NaN data with mean
newdata_BMI = data_BMI.fillna(int(data_BMI.mean()))

# select WTKG3 from dataset
data_WTK = df['WTKG3']
#fill the NaN data with mean
newdata_WTK = data_WTK.fillna(int(data_WTK.mean()))

# create new dataset
dataset = pd.DataFrame()
dataset['WTKG'] = newdata_WTK  # add WTK

# create class label according BMI
num = len(newdata_BMI)
Overweight = np.zeros((num, 1))
for i in range(num):
    if newdata_BMI[i] < 2500.0:
        Overweight[i] = -1
    else:
        Overweight[i] = 1  

dataset['Overweight'] = Overweight  # add Overweight label

dataset.to_csv(r'C:\Users\qiang\Desktop\2018 fall\5825\homework\5\wtkg.csv',index=False, sep=',', mode='a+')


