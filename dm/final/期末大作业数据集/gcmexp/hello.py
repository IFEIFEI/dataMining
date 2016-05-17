#!/usr/bin/env python

import numpy as np
from sklearn import cluster,metrics,datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
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

def decisionTree(dataTrain,featuresTrain,dataTest,featuresTest,filename='result'):
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
        print accuracy,recallu
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

def changeLabel(features):
    transformedFeatures=[]
    targetLabel=['CNS','Lung','Melanoma','Mesothelioma','Colorectal','Leukemia','Pancreas','Bladder','Renal']
    for label in features:
        if label in targetLabel:
            transformedFeatures.append(label)
        else:
            transformedFeatures.append('Unknown')
    return transformedFeatures

def chain1(dataTrain,featuresTrain,testVector):
    labels=['CNS','Leukemia','Mesothelioma','Renal']
    clf=SVC(probability=True,C=1.0,kernel='linear')
    percent=0.4
    col_num=(int)(dataTrain.shape[1]*percent)
    x,y=np.indices((dataTrain.shape[0],col_num))
    tempDateTrain=dataTrain[x,y]
    clf.fit(tempDateTrain,featuresTrain)
    resultFeature=clf.predict(testVector[:col_num])[0]
    print "chan1: "+resultFeature
    if resultFeature in labels:
        return resultFeature
    else:
        return 'None'

def chain2(dataTrain,featuresTrain,testVector):
    labels=['Bladder','Lung','Mesothelioma','Melanoma']
    clf=SVC(probability=True,C=1.0,kernel='rbf')
    percent=0.6
    col_num=(int)(dataTrain.shape[1]*percent)
    x,y=np.indices((dataTrain.shape[0],col_num))
    tempDateTrain=dataTrain[x,y]
    clf.fit(tempDateTrain,featuresTrain)
    resultFeature=clf.predict(testVector[:col_num])[0]
    print "chan2: "+resultFeature
    if resultFeature in labels:
        return resultFeature
    else:
        return 'None'

def chain3(dataTrain,featuresTrain,testVector):
    labels=['CNS','Lung','Melanoma','Mesothelioma']
    clf=SVC(probability=True,C=1.0,kernel='rbf')
    percent=0.1
    col_num=(int)(dataTrain.shape[1]*percent)
    x,y=np.indices((dataTrain.shape[0],col_num))
    tempDateTrain=dataTrain[x,y]
    clf.fit(tempDateTrain,featuresTrain)
    resultFeature=clf.predict(testVector[:col_num])[0]
    print "chan3: "+resultFeature
    if resultFeature in labels:
        return resultFeature
    else:
        return 'None'

def chain4(dataTrain,featuresTrain,testVector):
    labels=['Lung','Melanoma','Mesothelioma','Colorectal','Leukemia','Pancreas',]
    clf=SVC(probability=True,C=1.0,kernel='rbf')
    clf.fit(dataTrain,featuresTrain)
    resultFeature=clf.predict(testVector)[0]
    print "chan4: "+resultFeature
    if resultFeature in labels:
        return resultFeature
    else:
        return 'Unknown'
    print "chan4: "+resultFeature
    return resultFeature

def svmSvc(dataTrain,featuresTrain,dataTest,featuresTest,filename='result'):
    predict_features=[]
    for line in zip(dataTest,featuresTest):
        label=chain1(dataTrain,featuresTrain,line[0])
        if label != 'None':
            predict_features.append(label)
            print label+"::"+line[1]
            continue
        label=chain2(dataTrain,featuresTrain,line[0])
        if label != 'None':
            predict_features.append(label)
            print label+"::"+line[1]
            continue
        label=chain3(dataTrain,featuresTrain,line[0])
        if label != 'None':
            predict_features.append(label)
            print label+"::"+line[1]
            continue
        label=chain4(dataTrain,featuresTrain,line[0])
        predict_features.append(label)
        print label+"::"+line[1]
    accuracy=metrics.accuracy_score(featuresTest,predict_features)
    recallu=metrics.recall_score(featuresTest,predict_features)
    print accuracy,recallu




if __name__ == '__main__':
    dataTrain,featuresTrain,dataTest,featuresTest=initDate('GCM')
    dataTrain=preprocessing.scale(dataTrain)
    dataTest=preprocessing.scale(dataTest)
    featuresTrain=changeLabel(featuresTrain)
    featuresTest=changeLabel(featuresTest)
    print featuresTest

    print("\n\n$$$$$$$$$$$$$$$$$$$$$$$$$$GCM$$$$$$$$$$$$$$$$$$$$$$$$$$\n\n")
    print "######################SVMSVC######################"
    svmSvc(dataTrain,featuresTrain,dataTest,featuresTest,'GCM')
    print "######################SVMSVC######################"

    """
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$"+filename+"$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print "######################DecisionTree######################"
    decisionTree(dataTrain,featuresTrain,dataTest,featuresTest,filename)
    print "######################DecisionTree######################"
    """
