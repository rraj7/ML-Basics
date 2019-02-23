# CS 412 HW2 by Rishi Raj UIN - 663465793

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as mp
from pylab import show
import statistics
from sklearn import preprocessing
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

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


# TO find out the different values for degress of kernel

Error1 = []
CVal = []
data = [0.01,0.02,0.03,0.04,0.05,0.07,0.1,0.3,0.5,0.7,1,2,3,5,6,7,8,10,15,20,30,40,70,80,100]
for c in data:
    SVCModel = SVC(C =c, kernel = 'linear',gamma = 'auto')
    SVC1 = SVCModel.fit(st1,trainDigits)
    CrossFold = 1 - statistics.mean(cross_val_score(SVC1,st1,trainDigits,cv=10,scoring ='accuracy'))
    Error1.append(CrossFold)
    CVal.append(c)
mp.xscale('log')
mp.scatter(CVal,Error1)
mp.xlabel('log c')
mp.ylabel('Tenfold Cross Validations')
print("Min Error:",min(Error1))
Error1 = np.asarray(Error1)
LowestCrossValidationError = CVal[np.argmin(Error1)]
print ("Lowest C value:", LowestCrossValidationError )
show()



SVCModel = SVC(C =c, kernel = 'linear',gamma = 'auto')
SVC1 = SVCModel.fit(st1,trainDigits)

xPred = []
yPred = []
cPred = []
for xP in range(-100,100):
    xP = xP/100.0
    for yP in range(-100,100):
        yP = yP/100.0
        xPred.append(xP)
        yPred.append(yP)
        if(SVC1.predict([[xP,yP]])=="1.0"):
            cPred.append("r")
        else:
            cPred.append("b")

print('Creating the model')
mp.scatter(xPred,yPred,s=3,c=cPred,alpha=.2)
show()

#Finding the best degree

for d in [2, 5, 10, 20]:
    Error1 = []
    CVal= []
    for c in range (1,56):
         error = 2**c
         SVCModel =SVC(C=error, kernel ='poly', degree = d, gamma='auto')
         SVC1 = SVCModel.fit(st1,trainDigits)
         Error1.append(1 - statistics.mean(cross_val_score(SVC1,st1,trainDigits, cv=10,scoring='accuracy')))
         CVal.append(error)
    mp.xscale('log')
    mp.scatter(CVal,Error1)
    mp.xlabel('log c')
    mp.ylabel('Tenfold Cross Validations')
    LowestC = CVal[np.argmin(np.asarray(Error1))]
    print("For degree", d, "the lowest C value is", LowestC, "for Error =",min(Error1))
    show()


#Plot the decision region for the model with the best degree and the minimum error value
SVCModel =SVC(C=4314.398832739892, kernel ='poly', degree = 5, gamma='auto')
SVC1 = SVCModel.fit(st1,trainDigits)

xPred = []
yPred = []
cPred = []
for xP in range(-100,100):
    xP = xP/100.0
    for yP in range(-100,100):
        yP = yP/100.0
        xPred.append(xP)
        yPred.append(yP)
        if(SVC1.predict([[xP,yP]])=="1.0"):
            cPred.append("r")
        else:
            cPred.append("b")

print('Creating the model')
mp.scatter(xPred,yPred,s=3,c=cPred,alpha=.2)
show()



#Graduate Question - Overfitting
SVCModel =SVC(C=90, kernel ='poly', degree = 5, gamma='auto')
SVC1 = SVCModel.fit(st1,trainDigits)

xPred = []
yPred = []
cPred = []
for xP in range(-100,100):
    xP = xP/100.0
    for yP in range(-100,100):
        yP = yP/100.0
        xPred.append(xP)
        yPred.append(yP)
        if(SVC1.predict([[xP,yP]])=="1.0"):
            cPred.append("r")
        else:
            cPred.append("b")

print('Creating the model')
mp.scatter(xPred,yPred,s=3,c=cPred,alpha=.2)
show()

#Underfitting dor C = 150 d = 5

SVCModel =SVC(C=150, kernel ='poly', degree = 5, gamma='auto')
SVC1 = SVCModel.fit(st1,trainDigits)

xPred = []
yPred = []
cPred = []
for xP in range(-100,100):
    xP = xP/100.0
    for yP in range(-100,100):
        yP = yP/100.0
        xPred.append(xP)
        yPred.append(yP)
        if(SVC1.predict([[xP,yP]])=="1.0"):
            cPred.append("r")
        else:
            cPred.append("b")

print('Creating the model')
mp.scatter(xPred,yPred,s=3,c=cPred,alpha=.2)
show()

