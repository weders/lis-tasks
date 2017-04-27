import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler , PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
import seaborn as sns
import matplotlib.pylab as plt

def error_calculation(y_pred,y):
    RMSE = mean_squared_error(y,y_pred) ** 0.5
    print(RMSE)
    return


##########################################
# Loading Data Training data
##########################################

PATH_TO_TRAINING_DATA = 'data/train.csv'
PATH_TO_TEST_DATA = 'data/test.csv'
training_data = pd.read_csv(PATH_TO_TRAINING_DATA,sep=',', index_col = 0)
final_data = pd.read_csv(PATH_TO_TEST_DATA,sep=',',index_col = 0)

######################################################################
#preprocess data (extracting features ), split in test and train set
######################################################################

features_col = list(training_data)[1:]

#train data
y = training_data['y']
X = training_data[features_col]
#test data
final_X = final_data[features_col]

#split 
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 1)

transform = PolynomialFeatures(degree=3, interaction_only=False, include_bias=True)
X_train = transform.fit_transform(X_train)
X_test = transform.fit_transform(X_test)
print(X_test.shape)
lasso = Lasso(alpha=0.0001, fit_intercept=False, max_iter=100000)
lasso = lasso.fit(X_train,y_train)
y_test_pred = lasso.predict(X_test)

error_calculation(y_test_pred,y_test)



"""
#plot features vs target
sns.pairplot(training_data, x_vars=features_col,y_vars = 'y')
sns.plt.show()
"""


















