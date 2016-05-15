#all in one 
#SVM new implementation

import csv
import random
import math
import operator
import time
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
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
    
def euclideanDistance(instance1,instance2,length):
    distance=0
    for x in range(length):
        difference=instance1[x]-instance2[x]
        distance+= difference*difference 
    return math.sqrt(distance)

def neighbors(trainingSet,testcase,k):
    distances=[]
    length=len(testcase)-1
    for x in range(len(trainingSet)):
        dist=euclideanDistance(testcase,trainingSet[x],length)
        distances.append((trainingSet[x],dist))
    distances.sort(key=operator.itemgetter(1))
    kneighborslist=[]
    for x in range(k):
        kneighborslist.append(distances[x][0])
    return kneighborslist 

def getResponse(kneighborslist):
    DecisionVotes={}
    for x in range(len(kneighborslist)):
        decision=kneighborslist[x][-1]
        DecisionVotes[decision]=DecisionVotes.get(decision,0)+1
    return max(DecisionVotes.iteritems(), key=operator.itemgetter(1))[0]     
    
for nos in range(0,10):
    #getting the datasets ready     
    trainingSet=[]
    testSet=[]
    split=0.67
    loadDataset('spambase.data',split,trainingSet,testSet)
    
    x,y,z=[],[],[]    
    for i in range(len(trainingSet)-1000):
        x.append(trainingSet[i][:-1])
        y.append(trainingSet[i][-1])
    for i in range(len(testSet)):
        z.append(testSet[i][:-1]) 
    
    
    start_time=time.time()        
    #print "Simple SVM with rbf filter"
    clf = svm.SVC()
    #print clf
    clf.fit(x,y)
    
    predicted =clf.predict(np.float32(z))
    end_time=time.time()
    accuracy=getAccuracy(testSet,predicted)
    #print ('Accuracy:'+repr(accuracy)+"%")
    simulationtime=end_time-start_time
    #print "Simulation time: "+repr(simulationtime)+" seconds" 
    print "SVCwithrbf"+ " "+repr(accuracy)+" "+repr(simulationtime)
    
    #print " "    
    start_time=time.time()        
    #print "Linear SVM"
    clf = svm.LinearSVC()
    #print clf
    clf.fit(x,y)
    predicted =clf.predict(np.float32(z))
    end_time=time.time()
    accuracy=getAccuracy(testSet,predicted)
    #print ('Accuracy:'+repr(accuracy)+"%")
    simulationtime=end_time-start_time
    #print "Simulation time: "+repr(simulationtime)+" seconds"
    print "LinearSVM"+ " "+repr(accuracy)+" "+repr(simulationtime)
    
    #print " "
    start_time=time.time()        
    #print "Nu SVM"
    clf = svm.NuSVC()
    #print clf
    clf.fit(x,y)
    predicted =clf.predict(np.float32(z))
    end_time=time.time()
    accuracy=getAccuracy(testSet,predicted)
    #print ('Accuracy:'+repr(accuracy)+"%")
    simulationtime=end_time-start_time
    #print "Simulation time: "+repr(simulationtime)+" seconds"
    print "NuSVM"+ " "+repr(accuracy)+" "+repr(simulationtime)
  
    #print "" 
    #print "Naive Bayes"
    start_time=time.time()        
    model = GaussianNB()
    #print model
    model.fit(x,y)
    predicted= model.predict(z)
    end_time=time.time()
    accuracy=getAccuracy(testSet,predicted)
    #print ('Accuracy:'+repr(accuracy)+"%")
    simulationtime=end_time-start_time
    #print "Simulation time: "+repr(simulationtime)+" seconds"     
    print "NaiveBayes"+ " "+repr(accuracy)+" "+repr(simulationtime)
    
    #print " "
    start_time=time.time()
    #print "knn k=5"
    prediction=[]
    k=5
    for x in range(len(testSet)):
        kneighbors=neighbors(trainingSet,testSet[x],k) 
        prediction.append(getResponse(kneighbors))
    end_time=time.time()
    accuracy=getAccuracy(testSet,prediction)
    #print prediction
    #print ('Accuracy:'+repr(accuracy)+"%")
    simulationtime=end_time-start_time
    #print "Simulation time: "+repr(simulationtime)+" seconds"    
    print "KNN(k=5)"+ " "+repr(accuracy)+" "+repr(simulationtime)
    
    print " "+'Training set:'+repr(len(trainingSet))
    print " " +'Test Set:'+repr(len(testSet))
    
    print " "
    print " "
    
    
    