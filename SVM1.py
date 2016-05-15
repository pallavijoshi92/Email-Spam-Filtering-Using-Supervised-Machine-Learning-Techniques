#SVM new implementation

import csv
import random
import math
import operator
import time
from sklearn import svm
import numpy as np
def loadDataset(filename,split,trainingSet=[],testSet=[]):
    with open(filename,'rb') as csvfile:
        lines=csv.reader(csvfile)
        dataset=list(lines)
        for x in range(len(dataset)):
            for y in range(58):
                dataset[x][y]=float(dataset[x][y])
            if random.random()< split:
                trainingSet.append(dataset[x])           
            else:
                testSet.append(dataset[x])  
def getAccuracy(testSet,prediction):
    if len(testSet)!=len(prediction):
        print "some calculation error"
    correct=0    
    for i in range(len(testSet)):
        if testSet[i][-1]==prediction[i]:
            correct+=1     
    return (correct*1.0/len(testSet))*100                 
    
if __name__=="__main__":
    #getting the datasets ready     
    trainingSet=[]
    testSet=[]
    split=0.67
    loadDataset('spambase.data',split,trainingSet,testSet)
    print 'Training set:'+repr(len(trainingSet))
    print 'Test Set:'+repr(len(testSet)) 
    x,y,z=[],[],[]    
    for i in range(len(trainingSet)-1000):
        x.append(trainingSet[i][:-1])
        y.append(trainingSet[i][-1])
    for i in range(len(testSet)):
        z.append(testSet[i][:-1]) 
    start_time=time.time()        
    print "Simple SVM with rbf filter"
    clf = svm.SVC()
    print clf
    clf.fit(x,y)
    
    predicted =clf.predict(np.float32(z))
    end_time=time.time()
    accuracy=getAccuracy(testSet,predicted)
    print ('Accuracy:'+repr(accuracy)+"%")
    simulationtime=end_time-start_time
    print "Simulation time: "+repr(simulationtime)+" seconds" 

        
    start_time=time.time()        
    print "Linear SVM"
    clf = svm.LinearSVC()
    print clf
    clf.fit(x,y)
    predicted =clf.predict(np.float32(z))
    end_time=time.time()
    accuracy=getAccuracy(testSet,predicted)
    print ('Accuracy:'+repr(accuracy)+"%")
    simulationtime=end_time-start_time
    print "Simulation time: "+repr(simulationtime)+" seconds"
    
    start_time=time.time()        
    print "Nu SVM"
    clf = svm.NuSVC()
    print clf
    clf.fit(x,y)
    predicted =clf.predict(np.float32(z))
    end_time=time.time()
    accuracy=getAccuracy(testSet,predicted)
    print ('Accuracy:'+repr(accuracy)+"%")
    simulationtime=end_time-start_time
    print "Simulation time: "+repr(simulationtime)+" seconds"
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    