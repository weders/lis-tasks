import numpy as np
import pandas


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