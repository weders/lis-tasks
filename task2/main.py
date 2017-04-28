import pandas
import numpy as np

from sklearn import neighbors
from sklearn import neural_network
from sklearn import gaussian_process
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.manifold import LocallyLinearEmbedding

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



#######################################
# Routines
#######################################

def classification_error(y_predict,y_true):
    acc = accuracy_score(y_true,y_predict)
    return acc

def generate_output(test_id,test_predictions):
    test_id = np.reshape(test_id, (test_id.shape[0], 1))
    prediction_test = np.reshape(test_predictions, (test_predictions.shape[0], 1))
    output = np.hstack((test_id, prediction_test))

    for i in range(0, output.shape[0]):
        output[i, 0] = int(output[i, 0])

    test = pandas.DataFrame(data=output)
    test.to_csv('submission2.csv', header=['Id', 'y'], index=False)

    return




#######################################
# Loading Data
#######################################

PATH_TO_DATA = 'data/'
TRAINING_DATA = 'train.csv'
TEST_DATA = 'test.csv'

training_data = pandas.read_csv(PATH_TO_DATA+TRAINING_DATA,sep=',')
test_data = pandas.read_csv(PATH_TO_DATA+TEST_DATA,sep=',')


training_data = training_data.as_matrix()

training_x = training_data[:,2:]
training_labels = training_data[:,1]

test_data = test_data.as_matrix()
test_x = test_data[:,1:]
test_id = test_data[:,0]


#######################################
# Parameters
#######################################

SIZE_OF_TRAINING_SET = 600
NUMBER_OF_CLUSTERS= 3
NUMBER_OF_NEIGHBORS = 100




#######################################
# Split Data
#######################################


assert SIZE_OF_TRAINING_SET <= training_labels.shape[0]

validation_x = training_x[SIZE_OF_TRAINING_SET:,:]
training_x = training_x[:SIZE_OF_TRAINING_SET,:]

assert validation_x.shape[0]+training_x.shape[0] == training_data.shape[0]

validation_labels = training_labels[SIZE_OF_TRAINING_SET:]
training_labels = training_labels[:SIZE_OF_TRAINING_SET]

assert validation_labels.shape[0]+training_x.shape[0] == training_data.shape[0]

#######################################
# Data Analysis/Preprocessing
#######################################


lle = LocallyLinearEmbedding(n_components=3,n_neighbors=NUMBER_OF_NEIGHBORS)
embedded_x = lle.fit_transform(training_x)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

colors = ['b','y','r']

for i in range(0,training_x.shape[0]):
    ax.scatter(embedded_x[i,0],embedded_x[i,1],embedded_x[i,2],c=colors[int(training_labels[i])])

plt.show()




#######################################
# KMeans Implementation
#######################################

clf_kn = neighbors.KNeighborsClassifier(NUMBER_OF_NEIGHBORS,algorithm='kd_tree',leaf_size=20,metric='manhattan',p=1)
clf_kn.fit(training_x, training_labels)


clf_kn.fit(training_x,training_labels)
prediction_knneighbors = clf_kn.predict(validation_x)

acc_knn = classification_error(prediction_knneighbors,validation_labels)
print(acc_knn)

#######################################
# Neural Network Implementation
#######################################

#best_hidden_layer_size = 149
#acc_nn_best = 0.85

'''
for i in range(1,200):
    clf_nn = neural_network.MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=i, random_state=0)
    clf_nn.fit(training_x, training_labels)
    prediction_nn = clf_nn.predict(validation_x)

    acc_nn = classification_error(prediction_nn, validation_labels)
    if acc_nn > acc_nn_best:
        best_hidden_layer_size = i
        acc_nn_best = acc_nn

    print(i)

print(acc_nn_best, best_hidden_layer_size)

'''

'''
alpha = 1
#alpha_best = 2.9512665430652825e-05
acc_nn_best = 0.33

for i in range(1,100):
    clf_nn = neural_network.MLPClassifier(solver='lbfgs', alpha=alpha, hidden_layer_sizes=149, random_state=0)
    clf_nn.fit(training_x, training_labels)
    prediction_nn = clf_nn.predict(validation_x)

    acc_nn = classification_error(prediction_nn, validation_labels)
    if acc_nn > acc_nn_best:
        best_hidden_layer_size = i
        acc_nn_best = acc_nn
        print(acc_nn_best)

    alpha = 0.9**i
    print(i)

print(acc_nn_best,alpha)



number_of_random_states_best = 0
acc_nn_best = 0.33

for i in range(0,100):
    clf_nn = neural_network.MLPClassifier(solver='lbfgs', alpha=2.9512665430652825e-05, hidden_layer_sizes=149, random_state=i)
    clf_nn.fit(training_x, training_labels)
    prediction_nn = clf_nn.predict(validation_x)

    acc_nn = classification_error(prediction_nn, validation_labels)
    if acc_nn > acc_nn_best:
        best_random_state = i
        acc_nn_best = acc_nn
        print(acc_nn_best)
    print(i)

print(acc_nn_best, best_random_state)

'''



clf_nn = neural_network.MLPClassifier(solver='lbfgs', alpha=2.9512665430652825e-05, hidden_layer_sizes=149, random_state=9, activation='relu',learning_rate='adaptive',)
clf_nn.fit(training_x, training_labels)


clf_nn.fit(training_x,training_labels)
prediction_nn = clf_nn.predict(validation_x)

acc_nn = classification_error(prediction_nn, validation_labels)

print(acc_nn)

#######################################
# Gaussian Process Classifier
#######################################

'''
clf_gp = gaussian_process.GaussianProcessClassifier(optimizer='f_min_bfgs_b')
clf_gp.fit(training_x,training_labels)
prediction_gp = clf_gp.predict(validation_x)

acc_gp = classification_error(prediction_gp,validation_labels)
print(acc_gp)
'''

#######################################
# Support Vector Machine Classifier
#######################################

#polynomial = PolynomialFeatures(degree=2)
#training_features = polynomial.fit_transform(training_x)
#validation_features = polynomial.fit_transform(validation_x)

svc = SVC(C=6.0,kernel='linear',degree=5,gamma=0.01)
svc.fit(training_x,training_labels)

model = SelectFromModel(svc,prefit=True)
training_x_new = model.transform(training_x)
validation_x_new = model.transform(validation_x)

svc.fit(training_x_new,training_labels)
prediction_svc = svc.predict(validation_x_new)

acc_svc = classification_error(prediction_svc, validation_labels)
print(acc_svc)

#######################################
# Training for Test
#######################################

training_x = training_data[:,2:]
training_labels = training_data[:,1]



clf_nn_test = neural_network.MLPClassifier(solver='lbfgs',alpha=2.9512665430652825e-05,hidden_layer_sizes=149,random_state=9)
clf_nn_test.fit(training_x,training_labels)
prediction_test = clf_nn_test.predict(test_x)


#######################################
# Generate Output
#######################################

#generate_output(test_id,prediction_test)