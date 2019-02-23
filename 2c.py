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
#select the proportion of data to use for training
numTrain = int(len(features)*.2)
print('Creating training and testing set')
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

print('Generating 2D plot')
X = [(X1-min(X1))/(max(X1)-min(X1))]
X_print = 2*np.array(X) - 1
Y2 = np.array(Y1)
Y = [(Y2-min(Y2))/(max(Y2)-min(Y2))]
Y_print = 2*np.array(Y) - 1
mp.scatter(X_print,Y_print,s=3,c=colors,alpha=.2)
show()



st1= np.concatenate((X_print, Y_print),axis=0)
st1=np.swapaxes(st1, 0, 1)


#https://matplotlib.org/api/_as_gen/matplotlib.pyplot.scatter.html
#this just shows the points
#mp.scatter(X,Y,s=3,c=colors)
# show()

model = KNeighborsClassifier(n_neighbors=1)
model.fit(st1,trainDigits)

xPred = []
yPred = []
cPred = []
for xP in range(-100,100):
    xP = xP/100.0
    for yP in range(-100,100):
        yP = yP/100.0
        xPred.append(xP)
        yPred.append(yP)
        if(model.predict([[xP,yP]])=="1.0"):
            cPred.append("r")
        else:
            cPred.append("b")

print('Generating model')
mp.scatter(xPred,yPred,s=3,c=cPred,alpha=.2)
show()

#print(simpleTrain.shape)
#st1=np.concatenate(X_print.astype(int),Y_print.astype(int))
print('Mean errors: ')
print('Euclidean =' , 1 - statistics.mean(cross_val_score(model,st1,trainDigits,cv=10,scoring ='accuracy')))

model2 = KNeighborsClassifier(n_neighbors=1,metric='manhattan')
model2.fit(st1,trainDigits)
print('Manhattan =' , 1 - statistics.mean(cross_val_score(model2,st1,trainDigits,cv=10,scoring ='accuracy')))

model3 = KNeighborsClassifier(n_neighbors=1,metric='chebyshev')
model3.fit(st1,trainDigits)
print('Chebyshev =' , 1 - statistics.mean(cross_val_score(model3,st1,trainDigits,cv=10,scoring ='accuracy')))

knn=[]
Ecv=[]
StDev=[]
HighBound=[]

for m in range (0,25):
    knn.append(2*m+1)
    model4 = KNeighborsClassifier(n_neighbors=knn[m])
    Ecv.append(1 - statistics.mean(cross_val_score(model4,st1,trainDigits,cv=10,scoring ='accuracy')))
    StDev.append(statistics.stdev(cross_val_score(model4,st1,trainDigits,cv=10,scoring ='accuracy')))

print('95% CI working')
HighBound = (np.array(Ecv)+1.96*np.array(StDev))
print(StDev)
mp.plot(knn,HighBound)
show()
