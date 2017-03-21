import numpy as np
import sklearn 
from numpy import genfromtxt
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import csv

from math import fsum
import pandas

######################################################
# Preferences
######################################################

leastsquares = True

# least squares preferences

d = [1,1,1,1,1,1,1,1,1,1]
d = np.asarray(d)
d = np.reshape(d,(1,10))








gradientdescent = False
ridgeregression = True


######################################################
# Loading Data
######################################################


#load data from csv file to np.array
print("loading data...")
train_data = np.genfromtxt ('train.csv', delimiter=",")
test_data = np.genfromtxt ('test.csv', delimiter=",")
train_data = np.array(train_data)
test_data = np.array(test_data)
#test_data
x_test = test_data[1:2001,1:11]
#train and validate data
x_train = train_data[1:9000,2:12]
y_train = train_data[1:9000,1]
x_validate = train_data[9001:10000,2:12]
y_validate = train_data[9001:10000,1]

y_test = []

for i in x_test:
    y_test.append(fsum(i)/i.shape[0])

y_test = np.asarray(y_test)
test_Id = test_data[1:,0]
test_Id = np.reshape(test_Id,(test_Id.shape[0],1))
y_test = np.reshape(y_test,test_Id.shape)

output = np.hstack((test_Id,y_test))

outputFrame = pandas.DataFrame(data=output)
outputFrame.to_csv('submission.csv',header=['Id','y'],index=False)

