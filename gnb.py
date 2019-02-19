import numpy as np
import pandas as pd
import random
import math
from sklearn.metrics import classification_report

#split the data into training set and testing set
def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    testSet = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(testSet))
        trainSet.append(testSet.pop(index))
    return [trainSet, testSet]

#Calculate Prior
def calculatePrior(dataset):
    prior=np.ones(2)
    nrow = dataset.shape[0]
    prior[0] = np.sum(dataset[:,-1] == 0) / nrow
    prior[1] = np.sum(dataset[:,-1] == 1) / nrow
    return prior

#Learn Gaussian
def Learn_Gaussian(dataset):
    ncolumn = dataset.shape[1]-1
    mean_0 = np.zeros(ncolumn)
    std_0 = np.zeros(ncolumn)
    mean_1 = np.zeros(ncolumn)
    std_1= np.zeros(ncolumn)
    classes = dataset[:,-1]
    for c in np.arange(ncolumn):
        mean_0[c] = np.mean(dataset[:,c][classes == 0])
        std_0[c] = np.std(dataset[:,c][classes == 0])
        mean_1[c] = np.mean(dataset[:,c][classes == 1])
        std_1[c] = np.std(dataset[:,c][classes == 1])
    return mean_0,std_0,mean_1,std_1

#Calculate Likelyhood_Probability
def Likelyhood_Probability(dataset,mean_0,std_0,mean_1,std_1,p0,p1):
    [nrow,ncolumn]=dataset.shape
    g0=np.zeros(nrow*ncolumn).reshape((nrow,ncolumn))
    g1=np.zeros(nrow*ncolumn).reshape((nrow,ncolumn))
    for r in np.arange(nrow):
        for c in np.arange(ncolumn):
            g0[r][c] = - (pow(dataset[r,c] - mean_0[c],2) / (2 *pow (std_0[c],2))+np.log(p0)*0)
            g1[r][c] = - (pow(dataset[r,c] - mean_1[c],2) / (2 *pow (std_1[c],2))+np.log(p1)*0)
            #g0[r][c] = (-(1 / 2) * np.log(2 * math.pi) - np.log(std_0[c])) - (pow(dataset[r,c] - mean_0[c],2) / 2 *pow (std_0[c],2)+np.log(p0))
            #g1[r][c] = (-(1 / 2) * np.log(2 * math.pi) - np.log(std_1[c])) - (pow(dataset[r,c] - mean_1[c],2) / 2 *pow (std_1[c],2)+np.log(p1))
    return g0,g1

#Calculate Likelyhood
def Likelyhood(dataset,centernumber):
    r0 = 0
    r1 = 1
    nrow = dataset.shape[0]
    for r in np.arange(nrow):
        if dataset[r][0] > centernumber:
            r0 +=1
        else:
            r1 +=1
    likelyhood_0=r0/nrow
    likelyhood_1=r1/nrow
    return likelyhood_0,likelyhood_1

#Calculate Posterior
def calculatePoster(p,likelyhood):
    poster0 = p[0] * likelyhood[0]
    poster1 = p[1] * likelyhood[1]
    return poster0,poster1

def Predictions(p):
    bestLabel = None
    if p[0]>p[1]:
        bestLabel = 0
    else:
        bestLabel = 1
    
    return bestLabel

def getTestPredictions(testSet,g0,g1):
    predictions = []
    bestLabel = None
    for i in np.arange(testSet.shape[0]):
        if (g0[i]>g1[i]).all==True:   
            bestLabel = 1
        else:
            bestLabel = 0
        predictions.append(bestLabel)
    return predictions

def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(testSet.shape[0]):
        if testSet[i][-1] == predictions[i]:
            correct +=1
    return (correct/float(len(testSet))) * 100.0

if __name__ == '__main__':
    ##read data
    alldata = pd.read_csv(r'C:\Users\qiang\Desktop\2018 fall\5825\homework\2\creditcard.csv')
    
    #delete the two feature Time and Amount
    newdata=alldata.drop(['Time'], axis = 1)
    newdata=newdata.drop(['Amount'], axis = 1)
    
    #Converts data from the dataframe type to an array type
    alldataNP = np.array(alldata) 
    newdataNP = np.array(newdata) 
    
    # Get Time feature with classes
    newdata_Time = alldataNP[:,[0,-1]]
    # Get Amount feature with classes
    newdata_Amount = alldataNP[:,[-2,-1]]
    
    #Splite data into train and test
    splitRatio = 0.80
    trainSet,testSet = splitDataset(newdataNP, splitRatio)
    trainSet_Time,testSet_Time= splitDataset(newdata_Time, splitRatio)
    trainSet_Amount,testSet_Amount= splitDataset(newdata_Amount, splitRatio)
    print('Split dataset = {0} rows into trainset={1} and testset={2} rows'.format(len(newdata_Time), len(trainSet_Time), len(testSet_Time)))
    
    #Converts data from the dataframe type to an array type
    trainSetNP = np.array(trainSet)
    testSetNP = np.array(testSet)
    trainSet_TimeNP = np.array(trainSet_Time)
    testSet_TimeNP = np.array(testSet_Time)
    trainSet_AmountNP = np.array(trainSet_Amount)
    testSet_AmountNP = np.array(testSet_Amount)
    testSetNP_X = testSetNP[:,0:testSetNP.shape[1]-1]
    
    #Calculate prior posibility 
    p_V = calculatePrior(trainSetNP)

    #Calculate mean and variance using training set
    gaussian = Learn_Gaussian(trainSetNP)
    gaussian_mean_0=gaussian[0]
    gaussian_std_0=gaussian[1]
    gaussian_mean_1=gaussian[2]
    gaussian_std_1=gaussian[3]
    

    #calculate the g of the testing set
    g = Likelyhood_Probability(testSetNP_X, gaussian_mean_0, gaussian_std_0, gaussian_mean_1, gaussian_std_1, p_V[0], p_V[1])  
    g0 = g[0]
    g1 = g[1]
    
    #predict the testset
    predictions_test = getTestPredictions(testSetNP_X,g0, g1) 
    
    #calculate the accuracy of the prediction
    accuracy_test = getAccuracy(testSetNP, predictions_test)
    print('The accuracy of the prediction of the test data using gaussian naive bayes is {0} '.format(accuracy_test))
    
    #get true classes and predict classes 
    y_true = testSetNP[:,-1]
    y_pred = predictions_test
    
    #get the precision, recall and F1-score
    target_names = ['class 0', 'class 1']
    print('The precision, recall and F1-score using gaussian naive bayes is in the following table.')
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    #sort the  trainSet_TimeNP
    trainSet_TimeNP.sort(axis=0)

    #seperate the Time feature into Near and Far 
    len_traintime=int(0.5*trainSet_TimeNP.shape[0])  
    centernumber_Time=trainSet_TimeNP[len_traintime][0]
    #trainSet_TimeNP_Near = trainSet_TimeNP[0:len_traintime]
    #trainSet_TimeNP_Far = trainSet_TimeNP[len_traintime:]
    
    #len_testtime=int(0.5*testSet_TimeNP.shape[0])
    #testSet_TimeNP_Near = testSet_TimeNP[0:len_testtime]
    #testSet_TimeNP_Far = testSet_TimeNP[len_testtime:]
    
    #Calculate prior posibility 
    p_Time = calculatePrior(trainSet_TimeNP)

    #Calculate the likelyhood of the testSet
    likelyhood_Time = Likelyhood(testSet_TimeNP,centernumber_Time)   
    #print(likelyhood_Time)
   
    #Calculate the poster posibility
    poster_Time = calculatePoster(p_Time,likelyhood_Time)
    #print(poster_Time)
   
    #predict the testset
    predictions_Timetest = Predictions(poster_Time)
    #print(predictions_Timetest)
    
    #calculate the accuracy of the prediction
    #accuracy_test_Time = getAccuracy(testSet_TimeNP, predictions_Timetest)
    #print('The accuracy of the prediction of the test data of the feature Time is {0} '.format(accuracy_test))
    
    #get true classes and predict classes 
    y_true_Time = testSet_TimeNP[:,0]
    y_pred__Time = predictions_Timetest
    
    #get the precision, recall and F1-score
    target_names_Time = ['class 0', 'class 1']
    #print('The precision, recall and F1-score of the fearure Time is in the following table.')
    #print(classification_report(y_true_Time, y_pred__Time, target_names=target_names_Time))
    
    #sort the trainSet_AmountNP
    trainSet_AmountNP.sort(axis=0)
    
    #seperate the Amount feature into Near and Far 
    len_trainamount=int((1/3)*trainSet_AmountNP.shape[0])
    centernumber1_Amount=trainSet_AmountNP[len_trainamount][0]
    centernumber2_Amount=trainSet_AmountNP[2*len_trainamount][0]
    #trainSet_AmountNP_Low = trainSet_AmountNP[0:len_trainamount]
    #trainSet_AmountNP_Medium = trainSet_AmountNP[len_trainamount+1:2*len_trainamount]
    #trainSet_AmountNP_High = trainSet_AmountNP[2*len_trainamount+1:]
  
    #len_testamount=int((1/3)*testSet_AmountNP.shape[0])
    #testSet_AmountNP_Low = testSet_AmountNP[0:len_testamount]
    #testSet_AmountNP_Medium = testSet_AmountNP[len_testamount+1:2*len_testamount]
    #testSet_AmountNP_High = testSet_AmountNP[2*len_testamount+1:]
    
    #Calculate prior posibility 
    p_Amount = calculatePrior(trainSet_AmountNP)
      
    #Calculate the likelyhood of the testSet
    likelyhood_Amount = Likelyhood(testSet_AmountNP,centernumber1_Amount)

       
    #Calculate the poster posibility
    #poster_Low = calculatePoster(p_Amount[0], p_Amount[1], likelyhood_Low[0],likelyhood_Low[1])
    #poster_Medium = calculatePoster(p_Amount[0], p_Amount[1], likelyhood_Medium[0],likelyhood_Medium[1])
    #poster_High = calculatePoster(p_Amount[0], p_Amount[1], likelyhood_High[0],likelyhood_High[1])
    
    #predict the testset
    #predictions_Amounttest_Low = Predictions(poster_Low[0], poster_Low[1])
    #predictions_Amounttest_Medium = Predictions(poster_Medium[0], poster_Medium[1])
    #predictions_Amounttest_High = Predictions(poster_High[0], poster_High[1])    
    
    #calculate the accuracy of the prediction
    #accuracy_test_Amount = getAccuracy(testSet_AmountNP, predictions_Amounttest_Low)
    #print('The accuracy of the prediction of the test data of the feature Time is {0} '.format(accuracy_test))
    
    #get true classes and predict classes 
    #y_true_Amount = testSet_AmountNP[:,0]
    #y_pred__Amount = predictions_Amounttest_Low
    
    #get the precision, recall and F1-score
    target_names_Amount = ['class 0', 'class 1']
    #print('The precision, recall and F1-score of the fearure Amount is in the following table.')
    #print(classification_report(y_true_Amount, y_pred__Amount, target_names=target_names_Amount))

    
    