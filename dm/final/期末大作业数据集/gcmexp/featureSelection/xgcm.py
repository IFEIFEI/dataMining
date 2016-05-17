#!/usr/bin/env python

import numpy as np
from sklearn import cluster,metrics,datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from itertools import product

def readDateFromFile(filename,seperator=","):
    fd=open(filename)
    data2D=[]
    dataFeatures=[]
    dataFeatures=fd.readline()[:-1].split(seperator)
    for line in fd.readlines():
        data2D.append([float(lineAttr) for lineAttr in line[:-1].split(seperator)])
    data2D=np.transpose(data2D)
    return np.array(data2D),np.array(dataFeatures)

def initDate(filename):
    dataTrain,featuresTrain=readDateFromFile(filename+"_train.data")
    dataTest,featuresTest=readDateFromFile(filename+"_test.data")
    return dataTrain,featuresTrain,dataTest,featuresTest

def decisionTree(dataTrain,featuresTrain,dataTest,featuresTest,filename='result',percent=0.1):
    criterion=['gini','entropy']
    splitter=['best','random']
    max_features=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,None,"log2","sqrt"]
    max_accuracy=0.0
    recall_score=0.0
    best_clf=None
    for param in product(criterion,splitter,max_features):
        print("\n========================================================")
        clf=DecisionTreeClassifier(criterion=param[0],splitter=param[1],max_features=param[2])
        result=clf.fit(dataTrain,featuresTrain)
        print result
        print result.score(dataTest,featuresTest)
        resultFeatures=clf.predict(dataTest)
        accuracy=metrics.accuracy_score(featuresTest,resultFeatures)
        recallu=metrics.recall_score(featuresTest,resultFeatures)
        print accuracy,recallu,percent
        if accuracy > max_accuracy:
            max_accuracy=accuracy
            recall_score=recallu
            best_clf=result
            predict_features=resultFeatures
        print("\n========================================================")
    print ("\n Get result")
    print ("Best accuracy is %0.6f\nBest paramters is %s" % (max_accuracy,best_clf))
    fd = open("DT"+filename+".txt",'a')
    fd.write("Best accuracy and recall is %0.6f\t%0.6f\nBest paramters is %s" % (max_accuracy,recall_score,best_clf))
    fd.close()

def changeLabel2index(features):
    uniqueFeatures=np.unique(features)
    label_index_array=zip(uniqueFeatures,range(0,10))
    temp={}
    for label_index in label_index_array:
        temp[label_index[0]]=label_index[1]
    return [float(temp[f]) for f in features]


def svmSvc(dataTrain,featuresTrain,dataTest,featuresTest,filename='result',percent=0.1):
    C=[1.0]
    #kernel=['linear','poly','rbf','sigmoid']
    kernel=['linear','rbf','sigmoid']
    degree=[3]
    #gamma=['rbf','poly','sigmoid']
    gamma=['rbf']
    shrinking=[True]
    max_accuracy=0.0
    recall_score=0.0
    best_clf=None
    for param in product(C,kernel,degree,shrinking):
        print("\n========================================================")
        clf=SVC(probability=True,C=param[0],kernel=param[1],degree=param[2],shrinking=param[3])#,gamma=param[3],shrinking=param[4])
        result=clf.fit(dataTrain,featuresTrain)
        print result
        print result.score(dataTest,featuresTest)
        resultFeatures=clf.predict(dataTest)
        accuracy=metrics.accuracy_score(featuresTest,resultFeatures)
        recallu=metrics.recall_score(featuresTest,resultFeatures)
        #fbeta=metrics.fbeta_score(featuresTest,resultFeatures)
        #fscore=metrics.precison_recall_fscore_support(featuresTest,resultFeatures,beta=fbeta)
        print accuracy,recallu,percent
        print metrics.classification_report(featuresTest,resultFeatures)
        if accuracy > max_accuracy:
            max_accuracy=accuracy
            recall_score=recallu
            #max_fscore=fscore
            best_clf=result
        print("\n========================================================")
    print ("\n Get result")
    print ("Best accuracy is %0.6f\nBest paramters is %s" % (max_accuracy,best_clf))
    fd = open("SVC"+filename+".txt",'a')
    fd.write("\n\n"+str(dataTrain.shape[1])+"\n")
    fd.write("Best accuracy and recall is %0.6f\t%0.6f\nBest paramters is %s" % (max_accuracy,recall_score,best_clf))
    fd.close()




if __name__ == '__main__':
    dataTrain,featuresTrain,dataTest,featuresTest=initDate('GCM')
    dataTrain=preprocessing.scale(dataTrain)
    dataTest=preprocessing.scale(dataTest)
    sel=VarianceThreshold(threshold=(0.8*(1-0.8)))
    dataTrain=sel.fit_transform(dataTrain)
    features_num=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    print dataTrain.shape,featuresTrain.shape,dataTest.shape,featuresTest.shape
    for percent in features_num:
        x,y=np.indices((dataTrain.shape[0],(int)(dataTrain.shape[1]*percent)))
        tempDateTrain=dataTrain[x,y]
        x,y=np.indices((dataTest.shape[0],(int)(dataTest.shape[1]*percent)))
        tempDateTest=dataTest[x,y]
        print("\n\n$$$$$$$$$$$$$$$$$$$$$$$$$$GCM$$$$$$$$$$$$$$$$$$$$$$$$$$\n\n")
        print "######################SVMSVC######################"
        svmSvc(tempDateTrain,featuresTrain,tempDateTest,featuresTest,'GCM',percent)
        print "######################SVMSVC######################"

    """
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$"+filename+"$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print "######################DecisionTree######################"
    decisionTree(dataTrain,featuresTrain,dataTest,featuresTest,filename)
    print "######################DecisionTree######################"
    """
