import numpy as np
import sklearn 
from numpy import genfromtxt
from sklearn.metrics import mean_squared_error

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
x_train = train_data[1:4999,2:12]
y_train = train_data[1:4999,1]
x_validate = train_data[5000:10000,2:12]
y_validate = train_data[5000:10000,1]


######################################################
# Least Squares Implementation
######################################################

if leastsquares:

    # compute variance and covariance
    print("computing variance matrix and covariance matrix...")


    x_train_tra = x_train.transpose()
    var = x_train_tra.dot(x_train)
    covar = x_train_tra.dot(y_train)

    # compute weigths
    print("computing weights...")
    var_inv = np.linalg.inv(var)
    weights = var_inv.dot(covar)
    print(weights)
    # predict validate data
    print("calculating predictions...")
    prediction = x_validate.dot(weights)
    prediction = prediction.transpose()
    # rmse
    print("calculate RMSE..")
    RMSE = mean_squared_error(y_validate, prediction) ** 0.5
    print("RMSE:{0:0.15f}".format(RMSE))



######################################################
# Gradient Descent Implementation
######################################################

if gradientdescent:
    mean = np.mean(x_train,axis=0)
    mean = np.reshape(mean,(10,1))

    variance = (x_train.T.dot(x_train))

    eig_val,eig_vec = np.linalg.eigh(variance)
    l_min = eig_val[0]
    l_max = eig_val[-1]

    kappa = l_max/l_min
    precision = 10**-13

    t = int(kappa*precision)
    print(t)




    w_initial = np.ones((10,1))
    w = w_initial



######################################################
# Ridge Regression Implementation
######################################################

if ridgeregression:

    l = 1

    Id = np.identity(x_train.shape[1])

    variance = x_train.T.dot(x_train) + l*Id
    var_inv = np.linalg.inv(variance)

    covariance = x_train.T.dot(y_train)

    weights = var_inv.dot(covariance)






    print(weights)

    # predict validate data
    print("calculating predictions...")
    prediction = x_validate.dot(weights)
    prediction = prediction.transpose()

    print(y_validate.shape)
    y_validate = np.reshape(y_validate,(5000,1))
    prediction = np.reshape(prediction,(5000,1))

    # rmse
    print("calculate RMSE..")
    RMSE = mean_squared_error(y_validate, prediction) ** 0.5
    print("RMSE:{0:0.15f}".format(RMSE))


######################################################
# Result Output
######################################################

#predict test data and store into result.csv
pred_test = x_test.dot(weights)
pred_test = pred_test.transpose()
pred_test = pred_test.reshape((-1,1))
test_data = test_data[1:2001,0]
test_data = test_data.reshape((-1,1))
result = np.hstack((test_data,pred_test))
np.savetxt("result.csv", result, delimiter=",")


