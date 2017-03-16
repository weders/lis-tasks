import numpy as np
import sklearn 
from numpy import genfromtxt
from sklearn.metrics import mean_squared_error

######################################################
# Preferences
######################################################

leastsquares = True
gradientdescent = False


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
x_train = train_data[1:10000,2:12]
y_train = train_data[1:10000,1]
x_validate = train_data[5000:10000,2:12]
y_validate = train_data[5000:10000,1]


######################################################
# Least Squares Implementation
######################################################

#compute variance and covariance
print("computing variance matrix and covariance matrix...") 
x_train_tra = x_train.transpose()
var = x_train_tra.dot(x_train)
covar = x_train_tra.dot(y_train)

#compute weigths
print("computing weights...")
var_inv = np.linalg.inv(var)
weigths = var_inv.dot(covar)
print(weigths)
#predict validate data
print("calculating predictions...")
prediction = x_validate.dot(weigths)
prediction = prediction.transpose()
#rmse 
print("calculate RMSE..")
RMSE = mean_squared_error(y_validate, prediction)**0.5
print("RMSE:{0:0.15f}".format(RMSE))


######################################################
# Gradient Descent Implementation
######################################################







######################################################
# Result Output
######################################################

#predict test data and store into result.csv
pred_test = x_test.dot(weigths)
pred_test = pred_test.transpose()
pred_test = pred_test.reshape((-1,1))
test_data = test_data[1:2001,0]
test_data = test_data.reshape((-1,1))
result = np.hstack((test_data,pred_test))
np.savetxt("result.csv", result, delimiter=",")


