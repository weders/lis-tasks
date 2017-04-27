import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale
import optunity
import optunity.metrics

# sklearn algorithms
from sklearn import svm, linear_model, grid_search,kernel_ridge
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import *
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import *
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score , ShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict




##########################################
# Loading Data
##########################################

PATH_TO_TRAINING_DATA = 'data/train.csv'
PATH_TO_TEST_DATA = 'data/test.csv'

training_data = pd.read_csv(PATH_TO_TRAINING_DATA,sep=',')
test_data = pd.read_csv(PATH_TO_TEST_DATA,sep=',')


training_data = training_data.as_matrix()
test_data = test_data.as_matrix()

labels = training_data[:,1]
data = scale(training_data[:,2:])

test_data_Id = test_data[:,0]
test_data_x = test_data[:,1:]
test_data_x = scale(test_data_x)

test_data_Id = test_data_Id.astype(int)
test_data_Id = test_data_Id.T

data_train, data_validate, labels_train, labels_validate = train_test_split(data, labels, test_size=0.1)


shapeTestData = (2000,1)



##########################################
# sklearn regression methods
##########################################

def error_calculation(y_pred,y):
    RMSE = mean_squared_error(y,y_pred) ** 0.5
    print(RMSE)
    return

"""
poly_reg = svm.SVR(kernel = 'poly', C = 12 , epsilon = 0.1 , degree = 3)
poly_reg.fit(data_train,labels_train)
labels_poly = poly_reg.predict(data_validate)
error_calculation(labels_poly,labels_validate)

SGD = linear_model.SGDRegressor()
SGD.fit(data_train, labels_train)
labels_sgd = SGD.predict(data_validate)
error_calculation(labels_sgd,labels_validate)

"""
alpha = [2]
gamma = [0.1]
degree = [3]
parameters = {'alpha':alpha,'gamma':gamma,'degree':degree}
ridge = KernelRidge(kernel = 'poly')
ridge = GridSearchCV(ridge, parameters)
ridge.fit(data_train, labels_train)
labels_ridge = ridge.predict(data_validate)
error_calculation(labels_ridge,labels_validate)
predicted = cross_val_predict(ridge, data_validate, cv=5)
error_calculation(predicted,labels_validate)
print(ridge.score(data_validate,labels_validate))
"""
svr = svm.SVR(kernel = 'rbf',gamma = 0.005 , C = 100000, epsilon = 0.1)
#svr = GridSearchCV(svr, parameters)
svr.fit(data_train, labels_train)
labels_rbf= svr.predict(data_validate)
error_calculation(labels_rbf,labels_validate)
#print(svr.best_params_)


labels_total = (labels_rbf + labels_ridge)/2
error_calculation(labels_total,labels_validate)
"""




y_test = ridge.predict(test_data_x)






##########################################
# Result Output
##########################################

y_test = np.reshape(y_test,shapeTestData)

test_data_Id = np.reshape(test_data_Id,shapeTestData)

output1 = np.hstack((test_data_Id,y_test))



test = pd.DataFrame(data=output1)


test.to_csv('result.csv',header=['Id','y'],index=False)





