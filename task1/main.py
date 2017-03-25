import numpy as np
import pandas
from sklearn.metrics import mean_squared_error


##########################################
# Loading Data
##########################################

PATH_TO_TRAINING_DATA = 'data/train.csv'
PATH_TO_TEST_DATA = 'data/test.csv'

training_data = pandas.read_csv(PATH_TO_TRAINING_DATA,sep=',')
test_data = pandas.read_csv(PATH_TO_TEST_DATA,sep=',')


training_data = training_data.as_matrix()
test_data = test_data.as_matrix()

y_training = training_data[:,1]
x_training = training_data[:,2:]

test_data_Id = test_data[:,0]
test_data_x = test_data[:,1:]

test_data_Id = test_data_Id.astype(int)
test_data_Id = test_data_Id.T


##########################################
# Split Training Data
##########################################

fraction = 2 # enter fraction number for training data split

SIZE_OF_TRAINING_SET = int(training_data.shape[0]/fraction)

x_train = x_training[:SIZE_OF_TRAINING_SET,:]
y_train = y_training[:SIZE_OF_TRAINING_SET]

x_validation = x_training[SIZE_OF_TRAINING_SET:,:]
y_validation = y_training[SIZE_OF_TRAINING_SET:]


##########################################
# Routine Definitions
##########################################

def error_calculation(y_pred,y):
    RMSE = mean_squared_error(y,y_pred) ** 0.5
    print(RMSE)
    return




##########################################
# Preferences
##########################################

LLS = False # Linear Least Squares (order 1)
LLS_order = False # Linear Least Squares (order 2)
order_LLS = 2 # order of linear least squares


ridgeClosedForm = False
ridgeClosedForm_order = False
order_ridge = 2

ridgeGradientDescent = True

##########################################
# Linear Least Squares
##########################################

if LLS:
    # Training
    covariance = x_train.T.dot(x_train) # covariance matrix for linear least squares computation
    weights = np.linalg.inv(covariance).dot(x_train.T.dot(y_train)) # weights of linear least squares

    y_pred = x_validation.dot(weights) # prediction for validation set
    error_calculation(y_pred, y_validation) # error calculation for validation

if LLS_order:

    i = 2
    while i <= order_LLS:
        x_train = np.hstack((x_train,x_train**i))
        x_validation = np.hstack((x_validation,x_validation**i))
        i += 1


    covariance = x_train.T.dot(x_train) # covariance matrix for linear least squares computation
    weights = np.linalg.inv(covariance).dot(x_train.T.dot(y_train)) # weights of linear least squares


    y_pred = x_validation.dot(weights) # prediction for validation set
    error_calculation(y_pred, y_validation) # error calculation for validation


##########################################
# Ridge Regression Closed Form
##########################################

if ridgeClosedForm:

    mu = 10 # norm regularizer

    covariance = x_train.T.dot(x_train) + mu*np.identity(x_train.shape[1])
    weights = np.linalg.inv(covariance).dot(x_train.T.dot(y_train))

    y_pred = x_validation.dot(weights)  # prediction for validation set
    error_calculation(y_pred, y_validation)  # error calculation for validation


if ridgeClosedForm_order:

    mu = 175 # norm regularizer

    i = 2
    while i <= order_ridge:
        x_train = np.hstack((x_train, x_train ** i))
        x_validation = np.hstack((x_validation, x_validation ** i))
        i += 1

    covariance = x_train.T.dot(x_train) + mu*np.identity(x_train.shape[1])
    weights = np.linalg.inv(covariance).dot(x_train.T.dot(y_train))

    y_pred = x_validation.dot(weights)  # prediction for validation set
    error_calculation(y_pred, y_validation)  # error calculation for validation


##########################################
# Ridge Regression Gradient Descent
##########################################

if ridgeGradientDescent:
    eta = 0.001
    mu = 0.001
    n = x_train.shape[0]

    weights = np.ones((15, 1))  # initialization of weight vector

    covariance = x_train.T.dot(x_train)
    eigVals, eigVecs = np.linalg.eigh(covariance)

    l_max = eigVals[-1]
    l_min = eigVals[0]

    kappa = l_max / l_min

    NUMBER_OF_ITERATIONS = 10000
    print('Number of Iterations in Gradient Descent: ', NUMBER_OF_ITERATIONS)
    i = 1


    while i < NUMBER_OF_ITERATIONS:
        weights = (1 - eta * mu) * weights - eta / n * (covariance.dot(weights) - np.reshape(x_train.T.dot(y_train),weights.shape))
        i += 1
        print(i)

    y_pred = x_validation.dot(weights)  # prediction for validation set
    error_calculation(y_pred, y_validation)  # error calculation for validation



