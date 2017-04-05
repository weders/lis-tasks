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

SIZE_OF_TRAINING_SET = 800
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
las = True




#######################################################################
# Lasso regression methods
#######################################################################


if las:


# K-fold Crossvalidation - alpha value
##################################
    lasso = Lasso(normalize = True,max_iter=1e5)
    alphas = np.logspace(-4, -2, 20)
    scores = list()
    scores_std = list()
    a=False
    poly = PolynomialFeatures(degree=3)
    x_train2=poly.fit_transform(x_train)
    x_validation2=poly.fit_transform(x_validation)
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

# CV Lasso
##################################
    kf = KFold(n_splits=2)
    kf.get_n_splits(x_training)

    lasso = Lasso(normalize = True,max_iter=1e5)
    lasso.alpha=0.00483293

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
  
    

# result - whole dataset
##################################


    
    lasso.fit(x_train2,y_train)
    y_predict=lasso.predict(x_validation2)
    error_calculation(y_predict,y_validation)
    x_test_10=poly.fit_transform(test_data_x)
    y_test10=lasso.predict(x_test_10)

    



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
