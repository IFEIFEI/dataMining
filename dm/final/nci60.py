#!/usr/bin/env python

import numpy as np

def readDateFromFile(filename):
    fd=open(filename)
    data2D=[]
    dataFeatures=[]
    dataFeatures=fd.readline()[:-1].split("\t")
    for line in fd.readlines():
        data2D.append([float(lineAttr) for lineAttr in line[:-1].split("\t")])
    data2D=np.transpose(data2D)
    return np.array(data2D),np.array(dataFeatures)

def initDate():
    dataTrain,featuresTrain=readDateFromFile("nci60_train_m.txt")
    dataTest,featuresTest=readDateFromFile("nci60_test_m.txt")
    return dataTrain,featuresTrain,dataTest,featuresTest
    
if __name__ == '__main__':
    dataTrain,featuresTrain,dataTest,featuresTest=initDate()
    print dataTrain.shape,featuresTrain.shape 