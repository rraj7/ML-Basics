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



transformer = KernelPCA(n_components=2, kernel='poly', degree =3)
X_transformed = transformer.fit_transform(trainFeatures)


#Plot KPCA 
variance1 = np.var(X_transformed,axis =0)
variance_ratio1 = variance1/np.sum(variance1)
print(variance_ratio1)

for digits in range(0, len(trainDigits)):
	if (trainDigits[digits] =="1.0"):
		mp.scatter(X_transformed[digits,0], X_transformed[digits,1], color = "r")
	else: 
		mp.scatter(X_transformed[digits,0], X_transformed[digits, 1], color = "b")
mp.show()

##For Part 2

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


variance2 = np.var(st1,axis =0)
variance_ratio2 = variance2/np.sum(variance2)
print("For Hw1",variance_ratio2)

