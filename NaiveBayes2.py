#Naive Bayes Cross Validation Implementation

import csv
import random
import math
import operator
import time
import sklearn
from sklearn import svm
from sklearn import cross_validation
from sklearn.cross_validation import *
from sklearn.naive_bayes import GaussianNB
import sys
sys.stdout = open('NaiveBayesCrossValidation.txt', 'w')


#import numpy as np
def loadDataset(filename,split,data=[],target=[]):
    with open(filename,'rb') as csvfile:
        lines=csv.reader(csvfile)
        dataset=list(lines)
        for x in range(len(dataset)):
            for y in range(58):
                dataset[x][y]=float(dataset[x][y])
        for x in range(len(dataset)):
                data.append(dataset[x][:-1])           
                target.append(dataset[x][-1])  
def getAccuracy(testSet,prediction):
    if len(testSet)!=len(prediction):
        print "some calculation error"
    correct=0    
    for i in range(len(testSet)):
        if testSet[i]==prediction[i]:
            correct+=1     
    return (correct*1.0/len(testSet))*100                 
    
for cv in range(3,16):
    #getting the datasets ready
    #print('The scikit-learn version is {}.'.format(sklearn.__version__))    
    data=[]
    target=[]
    split=0.67
    loadDataset('spambase.data',split,data,target)
    print    '   Dataset:'+repr(len(data))
    #cv=10
    #print "   cross_validation:",cv
    start_time=time.time()        
    #print "SVM_with_rbf"
    model = GaussianNB()
    #print clf
    #predicted = cross_validation.cross_val_score(clf, data,target)
    predicted = cross_validation.cross_val_predict(model,data,target,cv=cv)
    end_time=time.time()
    accuracy=getAccuracy(target,predicted)
    #print ('Accuracy:'+repr(accuracy)+"%")
    simulationtime=end_time-start_time
    #print "Simulation time: "+repr(simulationtime)+" seconds" 
    print repr(cv)+" "+repr(accuracy)+" "+repr(simulationtime)   