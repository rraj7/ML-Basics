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

#model = KNeighborsClassifier(n_neighbors=1, metric='manhattan') #turns euclid into manhattan, put Chebychev to change
#print(simpleTrain.shape)
#st1=np.concatenate(X_print.astype(int),Y_print.astype(int))
#print(statistics.mean(cross_val_score(model,X1,trainDigits,cv=10,scoring ='accuracy')))
ecv=[]
knn=[]
for m in range(0,25):
    knn.append(2*m+1)
    model = KNeighborsClassifier(n_neighbors=knn[m], metric = 'euclidean') #metric = 'manhattan'turns euclid into manhattan, put Chebychev to change
#print(simpleTrain.shape)
#st1=np.concatenate(X_print.astype(int),Y_print.astype(int))
    ecv.append(1 - statistics.mean(cross_val_score(model,X1,trainDigits,cv=10,scoring ='accuracy')))
print(knn)
print(ecv)
mp.plot(knn,ecv)
mp.show()
