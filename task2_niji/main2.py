import numpy as np
import pandas
from sklearn.metrics import mean_squared_error

# sklearn algorithms
from sklearn.linear_model import (
    LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor)

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
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
a=False
b=True
if(a):
	#x_training = training_data[:,2:]
	x_train1 = training_data[:,2:8]
	x_train2 = training_data[:,11:] #14
	#x_train3 =training_data[:,16:]
	x_training =np.hstack((x_train1,x_train2))
	x_training = preprocessing.scale(x_training)
	test_data_Id = test_data[:,0]
	#test_data_x = test_data[:,1:]
	test1 = test_data[:,1:7]
	test2 = test_data[:,10:] #13
	#test3 = test_data[:,15:]
	test_data_x  =np.hstack((test1, test2))
	test_data_x = preprocessing.scale(test_data_x)

if(b):
	x_training = training_data[:,2:]
	x_training = preprocessing.scale(x_training)
	#x_train1 = x_training[:,0:6]
	#x_train2 = x_training[:,9:12]
	#x_train3 =  x_training[:,14:]
	#x_training =np.hstack((x_train1,x_train2,x_train3))
	test_data_Id = test_data[:,0]
	test_data_x = test_data[:,1:]
	test_data_x = preprocessing.scale(test_data_x)
	#test1 = test_data_x[:,0:6]
	#test2 = test_data_x[:,9:12]
	#test3 = test_data_x[:,14:]
	#test_data_x  =np.hstack((test1, test2, test3))

test_data_Id = test_data_Id.astype(int)
test_data_Id = test_data_Id.T

shapeTestData = (3000,1)


##########################################
# Split Training Data
##########################################
X_train, X_test, y_train, y_test = train_test_split(x_training, y_training, test_size=.05, random_state=42)

##########################################
# Routine Definitions
##########################################

def error_calculation(y_pred,y):
    acc = accuracy_score(y, y_pred)
    print(acc)
    return


##########################################
# MLPClassifier
##########################################
mlpc=MLPClassifier(hidden_layer_sizes=[100,50,10],alpha=3)


alphas = np.linspace(2, 4, 3)
scores = list()
scores_std = list()
a=False
if(a):
	n_folds = 20
    	for alpha in alphas:
   	 	mlpc.alpha = alpha
   	 	this_scores = cross_val_score(mlpc, x_training, y_training, cv=n_folds, n_jobs=1)
   	 	scores.append(np.mean(this_scores))
   	 	scores_std.append(np.std(this_scores))
	 	print('alpha')

    	scores, scores_std = np.array(scores), np.array(scores_std)

    
    	print(alphas)
    	print(scores)
    	print(max(scores))




mlpc=MLPClassifier(hidden_layer_sizes=[100,50,10],alpha=3)
this_scores=cross_val_score(mlpc, x_training, y_training, cv=20, n_jobs=1)
print(this_scores)
print(np.mean(this_scores))

#fit whole dataset
mlpc.fit(x_training,y_training)
#y_valid=mlpc.predict(X_test)
#error_calculation(y_valid,y_test)   
y_test10=mlpc.predict(test_data_x)

#plot = pandas.DataFrame(data=x_training)
#plt.matshow(plot.cov())
#plt.show()
##########################################
# Result Output
##########################################

y_test10 = np.reshape(y_test10,shapeTestData)
test_data_Id = np.reshape(test_data_Id,shapeTestData)
output1 = np.hstack((test_data_Id,y_test10))
test1 = pandas.DataFrame(data=output1)
test1.to_csv('result_test1.csv',header=['Id','y'],index=False)

