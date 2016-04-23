#!/usr/bin/env python

import time

from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import cross_validation
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def initDate(filename):
    fd=open(filename)
    data=[];
    features=[];
    for line in fd.readlines():
        if line[0]=="|":
            continue
        else:
            line=line[:-1]
            lineAttr=line.split(",")
            temp=[]
            for attr in lineAttr[:-1]:
                temp.append(float(attr))
            data.append(temp)
            features.append(lineAttr[-1].strip()[:-1])
    return np.array(data),np.array(features)

def initDate_leukemia(filename):
    fd=open(filename)
    data=[]
    for line in fd.readlines(): 
        #three spaces to seperate a row
        #two spaces to seperate a data in a row
        #for simple,just get them all and reshape the array
        lineAttr=(line[:-1]).split("  ")
        for attr in lineAttr:
            if attr.strip() != '':
                data.append(float(attr))
    return np.array(data).reshape(72,7130)
    
def processData(data):
    trainNum=int(len(data)*0.9)
    datasets=[]
    features=[]
    for row in data:
        datasets.append(row[1:])
        features.append(row[0])
    return datasets[:trainNum],features[:trainNum],datasets[trainNum:],features[trainNum:]
def makeTwoClassLabel(features):
    non_windows=['containers','tableware','headlamps']
    twoClassFeatures=[]
    for attr in features:
        if attr in non_windows:
            twoClassFeatures.append('Non-windows glass')
        else:
            twoClassFeatures.append('Window glass')
    return np.array(twoClassFeatures)

def getAccuracy(resultLabel,trueLabel):
    totalRight=0
    totalRecord=len(resultLabel)
    for line in (zip(resultLabel,trueLabel)):
        if line[0] == line[1]:
            totalRight+=1;
    return float(totalRight) / totalRecord

def simpleClf(clf,data,features,testData=None,testFeatures=None):
    print "======================================\n"
    print ("Start at: "+time.strftime("%H:%M:%S")+"\n")
    result=clf.fit(data,features)
    print result
    accuracy=0.0
    if testData!=None:
        testResult=clf.predict(testData)
        accuracy=getAccuracy(testResult,testFeatures)
    print ("End at: "+time.strftime("%H:%M:%S")+"\n")
    print "======================================\n"
    return result,accuracy

def mutipleClf(label_clfset,data,features,votingType='soft',weight=[],testData=None,testFeatures=None):
    flag=False
    if weight==[]:
        flag=True;
    print "======================================\n"
    print ("Start at: "+time.strftime("%H:%M:%S")+"\n")
    if votingType=='soft':  
        for label_clf in label_clfset:
            #use ten fold socore,set the cv to 10
            scores = cross_validation.cross_val_score(label_clf[1], data, features, cv=10)
            if flag:
                weight.append(scores.mean())
        eclf = VotingClassifier(estimators=label_clfset, voting=votingType, weights=weight)
        
    else:
        eclf = VotingClassifier(estimators=label_clfset, voting=votingType)
    result=eclf.fit(data,features)
    accuracy=0.0
    if testData!=None:
        testResult=eclf.predict(testData)
        accuracy=getAccuracy(testResult,testFeatures)   
    print ("End at: "+time.strftime("%H:%M:%S")+"\n")
    print "======================================\n"
    return result,accuracy


def process_glass_data(dataSet,features,testDataSet,testFeatures):
    clfSet=[]
    clf1 = DecisionTreeClassifier(max_depth=4)
    clf2 = KNeighborsClassifier(n_neighbors=3)
    clf3 = SVC(kernel='rbf', probability=True)
    clfSet.append(clf1)
    clfSet.append(clf2)
    clfSet.append(clf3)
    label_clfSet=zip(['DecisionTree','KNeighbors','SVC'],clfSet)
    print "--------------------------------"
    print "Use simple classfier"
    for label_clf in label_clfSet:
        result,accuracy=simpleClf(label_clf[1],dataSet,features,testData=testDataSet,testFeatures=testFeatures)
        print ("Use the classfier "+label_clf[0]+'\n')
        print "Accuracy : "+str(accuracy)+"\n"
        
    print "--------------------------------"
    print "Use mutiple classfier"
    print "Use the given weight of Simpson example 1 4 1"
    result,accuracy=mutipleClf(label_clfSet,dataSet,features,votingType='soft',weight=[2,4,2],testData=testDataSet,testFeatures=testFeatures)
    print ("Use the mutiple classfier by Simpon weight 1 4 1\n")
    print "Accuracy : "+str(accuracy)+"\n"
    print "--------------------------------"
    print "Use mutiple classfier"
    print "Use the given weight of Simple weight 1 2 1"
    result,accuracy=mutipleClf(label_clfSet,dataSet,features,votingType='soft',weight=[1,2,1],testData=testDataSet,testFeatures=testFeatures)
    print ("Use the mutiple classfier by Simple weight \n")
    print "Accuracy : "+str(accuracy)+"\n"
    print "--------------------------------"
    print "Use mutiple classfier"
    print "Use the the cross validation mean score to get weight"
    result,accuracy=mutipleClf(label_clfSet,dataSet,features,votingType='soft',weight=[],testData=testDataSet,testFeatures=testFeatures)
    print ("Use the mutiple classfier by cross_validation mean weight\n")
    print "Accuracy : "+str(accuracy)+"\n"
    print "--------------------------------"
    print "Use mutiple classfier"
    print "Use the the cross validation mean score to get weight and set the votingType to hard"
    result,accuracy=mutipleClf(label_clfSet,dataSet,features,votingType='hard',weight=[],testData=testDataSet,testFeatures=testFeatures)
    print ("Use the mutiple classfier by cross_validation mean weight\n")
    print "Accuracy : "+str(accuracy)+"\n"    

def get_glass_data():
    dataSet,features=initDate("glass.data")
    testDataSet,testFeatures=initDate("glass.test")
    twoClassFeatures=makeTwoClassLabel(features)
    testTwoClassFeatures=makeTwoClassLabel(testFeatures)
    print "###############################"
    print "Use two class to test data"
    process_glass_data(dataSet,features,testDataSet,testFeatures)
    print "###############################"
    print "Use all class to test data"
    process_glass_data(dataSet,twoClassFeatures,testDataSet,testTwoClassFeatures)      

def get_leukemia_data():
    originData=initDate_leukemia("LeukemiaDataSet3.dat")
    dataSet,features,testDataSet,testFeatures=processData(originData)
    clfSet=[]
    clf1 = DecisionTreeClassifier(max_depth=4)
    clf2 = KNeighborsClassifier(n_neighbors=3)
    clf3 = SVC(kernel='rbf', probability=True)
    clfSet.append(clf1)
    clfSet.append(clf2)
    clfSet.append(clf3)
    label_clfSet=zip(['DecisionTree','KNeighbors','SVC'],clfSet)
    print "--------------------------------"
    print "Use mutiple classfier"
    print "Use the the cross validation mean score to get weight"
    result,accuracy=mutipleClf(label_clfSet,dataSet,features,votingType='soft',weight=[],testData=testDataSet,testFeatures=testFeatures)
    print ("Use the mutiple classfier by cross_validation mean weight\n")
    print "Accuracy : "+str(accuracy)+"\n"
    print "--------------------------------"
    print "Use mutiple classfier"
    print "Use the the cross validation mean score to get weight and set the votingType to hard"
    result,accuracy=mutipleClf(label_clfSet,dataSet,features,votingType='hard',weight=[],testData=testDataSet,testFeatures=testFeatures)
    print ("Use the mutiple classfier by cross_validation mean weight\n")
    print "Accuracy : "+str(accuracy)+"\n"
    
if __name__=='__main__':
   print "\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
   print "Now show the glass data mining result"
   get_glass_data()
   print "\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
   print "Now show the leukemia data mining result"
   get_leukemia_data()
   
