# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 18:58:28 2018

@author: qiang
"""


import numpy as np
from math import log
import operator
import xport
import pandas as pd


def get_dataset():
    """
    BMI5: body mass index
    HTM4: height in meters
    PA1MIN_: variable for minutes of total physical activity per week
    
    """
    # select BMI5 from dataset
    data_BMI = df['_BMI5']
    #fill the NaN data with mean
    newdata_BMI = data_BMI.fillna(int(data_BMI.mean()))
    
    # select HTM4 from dataset
    data_HTM = df['HTM4']
    #fill the NaN data with mean
    newdata_HTM = data_HTM.fillna(int(data_HTM.mean()))
    
    # select WTKG3 from dataset
    data_WTK = df['WTKG3']
    #fill the NaN data with mean
    newdata_WTK = data_WTK.fillna(int(data_WTK.mean()))
    
    # select BMI5CAT from dataset
    # data_BMI5CAT = df['_BMI5CAT']
    # fill the NaN data with mean
    # newdata_BMI5CAT = data_BMI5CAT.fillna(2)
    
    # select PA1MIN from dataset
    data_PA1MIN = df['PA1MIN_']    
    #fill the NaN data with mean
    newdata_PA1MIN = data_PA1MIN.fillna(int(data_PA1MIN.mean()))
    
    # create new dataset
    dataset = pd.DataFrame()
    #dataset['BMI5CAT'] = newdata_BMI5CAT  # add BMI5CAT
    dataset['HTM'] = newdata_HTM  # add HTM
    dataset['WTKG'] = newdata_WTK  # add WTK
    dataset['PA1MIN'] = newdata_PA1MIN  # add PA1MIN
    
    # create class label according BMI
    num = len(newdata_BMI)
    Overweight = np.zeros((num, 1))
    for i in range(num):
        if newdata_BMI[i] < 2500.0:
            Overweight[i] = 0
        else:
            Overweight[i] = 1  
    dataset['Overweight'] = Overweight  # add Overweight label
    
    # create labels
    labels = ['HTM', 'WTKG', 'PA1MIN']
    
    return dataset,labels



def calc_entropy(dataSet):
    """
    calculate the entropy 
    """
    num = len(dataSet)
    labelCounts = {}
    
    # get the the labelCounts
    for featVec in dataSet: 
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    
    # calculate the entropy  
    entropy = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/num
        entropy -= prob * log(prob,2) 
    return entropy

# entropytest =  calc_entropy(t)
# entropytest


def split_data_set(data_set, index, value, part=0):
    """
    split the data set by attribute
    index:the index of the partition attribute
    value:the value of the partition attribute
    part: 0 represents the data set to the left of the partition point, 
          1 represents the data set to the right of the partition point
    """
    # save the subdataset
    res_data_set = []
    
    for entry in data_set:
        # find the data set to the left of the partition point
        if part == 0 and float(entry[index])<= value: #求划分点左侧的数据集
            reduced_entry = entry[:index]
        # after partitioning, the value of the index column in the data is removed
            reduced_entry.extend(entry[index + 1:]) 
            res_data_set.append(reduced_entry)
        # find the data set to the right of the partition point
        if part ==1 and float(entry[index])> value: 
            reduced_entry = entry[:index]
            reduced_entry.extend(entry[index + 1:])
            res_data_set.append(reduced_entry)
    return res_data_set

# splittest = split_data_set(t, 0, 165.0,part=0)
# splittest


def attribute_selection_method(data_set):
    """
     

    """
    # number of the attributes
    num_attributes = len(data_set[0])-1 
    
    # calculate the dataset entropy
    info_D = calc_entropy(data_set)  
    
    # max information gain ratio 
    max_gain_rate = 0.0 
    
    # get the best attribute index and best split point
    best_attribute_index = -1
    best_split_point = None

    for i in range(num_attributes):
        # get attribute list
        attribute_list = [entry[i] for entry in data_set]
        # calculate the information gain of attribute A to data set D
        info_A_D = 0.0
        # calculate the entropy of the data set D with respect to the value of attribute A
        split_info_D = 0.0  
        # sort all the discrete values under this attribute
        attribute_list = np.sort(attribute_list)
        # Set is used to eliminate the same value 
        temp_set = set(attribute_list) 
        attribute_list = [attr for attr in temp_set]
        split_points = []
        '''
        The median point between two adjacent values is used as the partition point 
        to calculate the information gain ratio, and the partition point corresponding 
        to the maximum gain ratio is the optimal partition point
        '''
        for index in range(len(attribute_list) - 1):
            # find the partition points
            split_points.append((float(attribute_list[index]) + float(attribute_list[index + 1])) / 2)
        # tranverse the partition points  
        for split_point in split_points:
            info_A_D = 0.0
            split_info_D = 0.0
            # The optimal partition point splits the data into two pieces, 
            # so two pieces of data can be obtained after two cycles
            for part in range(2): 
                sub_data_set = split_data_set(data_set, i, split_point, part)
                prob = len(sub_data_set) / float(len(data_set))
                info_A_D += prob * calc_entropy(sub_data_set)
                split_info_D -= prob * log(prob, 2) 
                # dealing with the 0 entropy
                if split_info_D==0:
                    split_info_D+=1
                # calculate the information gain ratio    
                gain_rate = (info_D - info_A_D) / split_info_D
                #select the max information gain rate
                if gain_rate > max_gain_rate:
                    max_gain_rate = gain_rate
                    best_split_point = split_point
                    best_attribute_index = i
    return best_attribute_index,best_split_point

#attribute_selection_method_test = attribute_selection_method(t)
#attribute_selection_method_test


def most_voted_attribute(label_list):
    """
    Majority vote: returns the largest number of classes in the tag list
    """
    label_nums = {}
    for label in label_list:
        if label in label_nums.keys():
            label_nums[label] += 1
        else:
            label_nums[label] = 1
    sorted_label_nums = sorted(label_nums.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_label_nums[0][0]


def generate_decision_tree(data_set ,attribute_label):
    """
    generate decision tree
    data_set: training set
    attribute_label: attibute labels
    the decision tree is represented by a dictionary structure, 
    recursive generation
    """
    # get tge label list
    label_list = [entry[-1] for entry in data_set]
    
    # if all data belongs to the same class, the class is returned
    if label_list.count(label_list[0]) == len(label_list):
        return label_list[0]
    
    # if the data has no attribute value data, return the class 
    # that appears most in the data as a class
    if len(data_set[0]) == 1: 
        return most_voted_attribute(label_list)
    
    # get tge best arribute index and best split point
    best_attribute_index, best_split_point = attribute_selection_method(data_set)
    best_attribute = attribute_label[best_attribute_index]
    
    # create decision tree
    decision_tree = { best_attribute:{}}
    
    # after finding the best partition property,
    # remove it from the attribute list
    del(attribute_label[best_attribute_index]) 

    # the calculated best partition point splits the data set in two pieces
    # one is <= best partition point, another is >best partition point
    sub_labels = attribute_label[:]
    decision_tree[best_attribute]["<="+str(best_split_point)] = generate_decision_tree(
        split_data_set(data_set, best_attribute_index, best_split_point, 0), sub_labels)
    sub_labels = attribute_label[:]
    decision_tree[best_attribute][">" + str(best_split_point)] = generate_decision_tree(
            split_data_set(data_set, best_attribute_index, best_split_point, 1), sub_labels)
    return decision_tree

#generate_test = generate_decision_tree(t ,labels)
#generate_test
#labels = ['HTM', 'WTKG', 'PA1MIN']  


def decision_tree_predict(decision_tree, attribute_labels, one_test_data):
    """
    A testing set is predicted by recursively predicting the label of the data
    decision_tree: dictionary structure decision tree
    attribute_labels: the attribute labels of testing set
    one_test_data：one data in the testing set
    return the result label
    """
    
    first_key = list(decision_tree.keys())[0]
    second_dic = decision_tree[first_key]
    attribute_index = attribute_labels.index(first_key)
    result_label = None
    
    for key in second_dic.keys(): 
        # two situations, one is <=, another is >
        if key[0] == '<':
            value = float(key[2:])
            if float(one_test_data[attribute_index])<= value:
                if type(second_dic[key]).__name__ =='dict':
                    result_label = decision_tree_predict(second_dic[key], attribute_labels, one_test_data)
                else:
                    result_label = second_dic[key]
        elif key[0] == '>':
            value = float(key[1:])
            if float(one_test_data[attribute_index]) > value:
                if type(second_dic[key]).__name__ == 'dict':
                    result_label = decision_tree_predict(second_dic[key], attribute_labels, one_test_data)
                else:
                    result_label = second_dic[key]
    return result_label


if __name__ == '__main__':
    
    # import the dataset
    path = r"C:\Users\qiang\Desktop\2018 fall\5825\homework\project\LLCP2017.XPT"
    with open(path,'rb') as f:
        df = xport.to_dataframe(f)
    
    # create dataset and labels
    dataset,labels = get_dataset() 
    
    # get 8000 samples from dataset
    train_data = dataset.sample(n=8000)

    # get 2000 samples from dataset
    test_data = dataset.sample(n=2000)

    # convert train_set and test_set to list
    train_data_array = train_data.values
    test_data_array = test_data.values
    train_data_list = train_data_array.tolist()
    test_data_list = test_data_array.tolist()
    
    # generate the decision trees
    decision_tree = generate_decision_tree(train_data_list ,labels)
    # decision_tree
    
    labels = ['HTM', 'WTKG', 'PA1MIN']   

    count = 0
    # calculate the accuracy according testing set
    for one_test_data in test_data_list:
        if decision_tree_predict(decision_tree,labels,one_test_data) == one_test_data[-1]:
            count+=1
    accuracy = count/len(test_data) 
    print(decision_tree)
    print('Training set size:8000，Testing set size:2000，Accuracy:%.1f%%'%(100*accuracy))
    
    
