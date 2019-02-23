# CS 412 HW2 by Rishi Raj UIN - 663465793

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as mp
from pylab import show
import statistics
from sklearn import preprocessing
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import KernelPCA
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression

data = np.loadtxt("data.csv")


#shuffle the data and select training and test data
np.random.seed(100)
np.random.shuffle(data)


features = []
digits = []


for row in data:
    if(row[0]==1 or row[0]==5):
        features.append(row[1:])
        digits.append(str(row[0]))
print('test')

#select the proportion of data to use for training
numTrain = int(len(features)*.2)

trainFeatures = features[:numTrain]
testFeatures = features[numTrain:]
trainDigits = digits[:numTrain]
testDigits = digits[numTrain:]

#create the model
#https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html


X1 = []
colors = []
for index in range(len(trainFeatures)):
    X1.append(trainFeatures[index][:])
    if(trainDigits[index]=="1.0"):
        colors.append("r")
    else:
        colors.append("b")




X1 = []
Y1 = []
simpleTrain = []
colors = []
for index in range(len(trainFeatures)):
    X1.append(statistics.mean(trainFeatures[index][:]))
    Y1.append(statistics.stdev(trainFeatures[index][:]))
    simpleTrain.append([statistics.mean(trainFeatures[index][:]),statistics.stdev(trainFeatures[index][:])])
    if(trainDigits[index]=="1.0"):
        colors.append("r")
    else:
        colors.append("b")

X = [(X1-min(X1))/(max(X1)-min(X1))]
X_print = 2*np.array(X) - 1
Y2 = np.array(Y1)
Y = [(Y2-min(Y2))/(max(Y2)-min(Y2))]
Y_print = 2*np.array(Y) - 1
mp.scatter(X_print,Y_print,s=3,c=colors,alpha=.2)
show()



st1= np.concatenate((X_print, Y_print),axis=0)
st1=np.swapaxes(st1, 0, 1)


#Logistic Regression 

LogisticModel = LogisticRegression(penalty ='l2',C=0.01)
LogisticModel.fit(st1,trainDigits)

xPred = []
yPred = []
cPred = []
for xP in range(-100,100):
    xP = xP/100.0
    for yP in range(-100,100):
        yP = yP/100.0
        xPred.append(xP)
        yPred.append(yP)
        if(LogisticModel.predict([[xP,yP]])=="1.0"):
            cPred.append("r")
        else:
            cPred.append("b")

mp.scatter(xPred,yPred,s=3,c=cPred,alpha=.5)
show()


#for c = 2.0
LogisticModel = LogisticRegression(penalty ='l2',C=2.0)
LogisticModel.fit(st1,trainDigits)

xPred = []
yPred = []
cPred = []
for xP in range(-100,100):
    xP = xP/100.0
    for yP in range(-100,100):
        yP = yP/100.0
        xPred.append(xP)
        yPred.append(yP)
        if(LogisticModel.predict([[xP,yP]])=="1.0"):
            cPred.append("r")
        else:
            cPred.append("b")

mp.scatter(xPred,yPred,s=3,c=cPred,alpha=.5)
show()


#For L1 Regularization
LogisticModel = LogisticRegression(penalty ='l1',C=0.01)
LogisticModel.fit(st1,trainDigits)

xPred = []
yPred = []
cPred = []
for xP in range(-100,100):
    xP = xP/100.0
    for yP in range(-100,100):
        yP = yP/100.0
        xPred.append(xP)
        yPred.append(yP)
        if(LogisticModel.predict([[xP,yP]])=="1.0"):
            cPred.append("r")
        else:
            cPred.append("b")

mp.scatter(xPred,yPred,s=3,c=cPred,alpha=.5)
show()

#For L1 c = 2.0
LogisticModel = LogisticRegression(penalty ='l2',C=2.0)
LogisticModel.fit(st1,trainDigits)

xPred = []
yPred = []
cPred = []
for xP in range(-100,100):
    xP = xP/100.0
    for yP in range(-100,100):
        yP = yP/100.0
        xPred.append(xP)
        yPred.append(yP)
        if(LogisticModel.predict([[xP,yP]])=="1.0"):
            cPred.append("r")
        else:
            cPred.append("b")

mp.scatter(xPred,yPred,s=3,c=cPred,alpha=.5)
show()
