import numpy as np
import sklearn 
from numpy import genfromtxt


#load data from csv file to np.array to x_train and y_train
print("loading data...")
train_data = np.genfromtxt ('train.csv', delimiter=",")
test_data = np.genfromtxt ('test.csv', delimiter=",")

train_data = np.array(train_data)
train_data.reshape((10001,12))
test_data = np.array(test_data)
x_test = test_data[1:2001,1:11]
x_train = train_data[1:10000,2:12]
y_train = train_data[1:10000,1]

#compute variance and covariance
print("computing variance matrix and covariance matrix...") 
x_train_tra = x_train.transpose()
var = x_train_tra.dot(x_train)
covar = x_train_tra.dot(y_train)
#compute weigths
print("computing weights...")
var_inv = np.linalg.inv(var)
weigths = var_inv.dot(covar)
#predict x_test
print("calculating y_test...")
y_test = x_test.dot(weigths)
y_test = y_test.transpose()
print(y_test)





