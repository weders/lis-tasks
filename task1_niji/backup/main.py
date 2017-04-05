import numpy as np
import pandas
from sklearn.metrics import mean_squared_error

# sklearn algorithms
from sklearn.linear_model import (
    LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor)

from sklearn import linear_model
from sklearn.svm import SVR,NuSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import KFold
from sklearn import linear_model


from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.preprocessing import PolynomialFeatures

import numpy as np
import matplotlib.pyplot as plt

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

shapeTestData = (2000,1)


##########################################
# Split Training Data
##########################################

fraction = 2 # enter fraction number for training data split

SIZE_OF_TRAINING_SET = 899
#x_training=preprocessing.scale(x_training)
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

ridgeGradientDescent = False
skl = True

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



##########################################
# sklearn regression methods
##########################################


if skl:
    c1=500
    c2=1000
    h=0.01
    K1=rbf_kernel(x_train,x_train,h)
    K2=linear_kernel(x_train)
    K3=c1*K1+c2*K2
    a=K3.shape[1]
    unity=np.eye(a)
    param_lambda=1;
    #print(param_lambda*unity)
    b=K3+param_lambda*unity
    inv_b=np.linalg.inv(b)
    opt_weights=inv_b.dot(y_train)
   

    K_rbf=rbf_kernel(x_train,x_validation,h)
    K_linear=linear_kernel(x_train,x_validation)
    
    K_valid=c1*K_rbf+c2*K_linear
    print(K_rbf.shape)
    print(K_linear.shape)
    y_predict1=K_valid.T.dot(opt_weights)
    

    #estimator1 = SVR(kernel='precomputed', C=1e4, gamma=0.01)
    
    #estimator1 = SVR(kernel='precomputed')
    #estimator1.fit(K3,y_train)

    

    #K1=rbf_kernel(x_validation,x_validation,1e4)
    #K2=linear_kernel(x_validation)
    #K3=K1+K2
    #y_predict1 = estimator1.predict(K3)
    error_calculation(y_predict1,y_validation)
    

    K_rbf=rbf_kernel(x_train,test_data_x,h)
    K_linear=linear_kernel(x_train,test_data_x)
    K_valid=c1*K_rbf+c2*K_linear
    y_test1=K_valid.T.dot(opt_weights)
    #K1=rbf_kernel(test_data_x,test_data_x,1e4)
    #K2=linear_kernel(test_data_x)
    #K3=K1+K2
    #y_test1 = estimator1.predict(K3)


    #estimator2 = SVR(kernel='poly',C=1e3, degree=2)
    #estimator2.fit(x_train, y_train)
    #y_predict2 = estimator2.predict(x_validation)

    #error_calculation(y_predict2, y_validation)

    #y_test2 = estimator2.predict(test_data_x)
  
    number=0
    print('hallo')
    kf = KFold(n_splits=2)
    kf.get_n_splits(x_training)
    
    poly = PolynomialFeatures(degree=3)
    x_train2=poly.fit_transform(x_train)
    x_test_10=poly.fit_transform(test_data_x)
    x_validation2=poly.fit_transform(x_validation)
    lasso = Lasso(normalize = True,max_iter=1e5)
    lasso.alpha=0.00483293
########


    print('poly')
    lasso.fit(x_train2,y_train)

    y_predict=lasso.predict(x_validation2)
    print(x_validation2.shape)
    print(y_predict.shape)
    print(y_validation.shape)
    error_calculation(y_predict,y_validation)
    
    y_test10=lasso.predict(x_test_10)
########
    
#y_test3=lasso.predict(x_test)
    b=False
    if(b):
    	for train_index, test_index in kf.split(x_training):
       		x_train, x_validation = x_training[train_index], x_training[test_index]
       		y_train, y_validation = y_training[train_index], y_training[test_index]
       		x_train=poly.fit_transform(x_train)
       		x_validation=poly.fit_transform(x_validation)
       		lasso.fit(x_train,y_train)
       		y_predict=lasso.predict(x_validation)
       		error_calculation(y_predict,y_validation) 
    	print('fertig')
    
    
	


    lasso = Lasso(normalize = True,max_iter=1e5)
    alphas = np.logspace(-4, -2, 20)
    scores = list()
    scores_std = list()
###########################################################
    a=False
    if(a):
    	n_folds = 5
    	
    	
    	for alpha in alphas:
   	 	lasso.alpha = alpha
   	 	this_scores = cross_val_score(lasso, x_train2, y_train, cv=n_folds, n_jobs=1)
   	 	scores.append(np.mean(this_scores))
   	 	scores_std.append(np.std(this_scores))
	 	print('alpha')

    	scores, scores_std = np.array(scores), np.array(scores_std)

    
    	print(alphas)
    	print(scores)
    	print(max(scores))
##########################################################  


##########################################
# Result Output
##########################################

y_test10 = np.reshape(y_test10,shapeTestData)
#y_test2 = np.reshape(y_test2,shapeTestData)
#y_test3 = np.reshape(y_test3,shapeTestData)

test_data_Id = np.reshape(test_data_Id,shapeTestData)

output1 = np.hstack((test_data_Id,y_test10))
#output2 = np.hstack((test_data_Id,y_test2))
#output3 = np.hstack((test_data_Id,y_test3))


test1 = pandas.DataFrame(data=output1)
#test2 = pandas.DataFrame(data=output2)
#test3 = pandas.DataFrame(data=output3)

test1.to_csv('result_test1.csv',header=['Id','y'],index=False)
#test2.to_csv('result_test2.csv',header=['Id','y'],index=False)
#test3.to_csv('result_test3.csv',header=['Id','y'],index=False)
