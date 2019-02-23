#From the console, run the following
#pip install numpy
#pip install scipy
#pip install scikit-learn
#pip install matplotlib

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as mp
from pylab import show
import statistics
from sklearn import preprocessing
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
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

#https://matplotlib.org/api/_as_gen/matplotlib.pyplot.scatter.html
#this just shows the points
#mp.scatter(X,Y,s=3,c=colors)
# show()



Error1 = []
CVal = []
data = [0.01,0.02,0.03,0.04,0.05,0.07,0.1,0.3,0.5,0.7,1,2,3,5,6,7,8,10,15,20,30,40,70,80,100]
for c in data:
    SVCModel = SVC(C =c, kernel = 'linear',gamma = 'auto')
    SVC1 = SVCModel.fit(X1,trainDigits)
    CrossFold = 1 - statistics.mean(cross_val_score(SVC1,X1,trainDigits,cv=10,scoring ='accuracy'))

    Error1.append(CrossFold)
    CVal.append(c)
mp.xscale('log')
mp.scatter(CVal,Error1)
show()

