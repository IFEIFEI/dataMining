#!/usr/bin/env python

from sklearn import datasets
from sklearn import svm

def initDate():
    fd=open("glass.data")
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
            features.append(lineAttr[-1].strip())
    return data,features
    
if __name__=='__main__':
    dataSets,features=initDate()
    clfSet=[]
    clfSet.append(svm.LinearSVC())
    clfSet.append(DecisionTreeClassifier(max_depth=4))
    clf = svm.LinearSVC()
    svmResult=clf.fit(dataSets,features)
    print clf.predict(dataSets[0])
    print svmResult
    