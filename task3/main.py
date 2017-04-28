###############################################
# Imports
###############################################

import pandas
import numpy as np
import tensorflow as tf
import NeuralNetwork as NN
import os


###############################################
# Loading/Preprocessing Data
###############################################

PATH_TO_TRAINING_DATA = 'data/train.h5'
PATH_TO_TEST_DATA = 'data/test.h5'

train = pandas.read_hdf(PATH_TO_TRAINING_DATA, "train")
test = pandas.read_hdf(PATH_TO_TEST_DATA, "test")

data_train = train.as_matrix()
data_test = test.as_matrix()


SIZE_OF_TRAINING_SET = 30000

###############################################
# Visualization
###############################################


assert SIZE_OF_TRAINING_SET < data_train.shape[0]

training_data = data_train[:SIZE_OF_TRAINING_SET, :]
validation_data = data_train[SIZE_OF_TRAINING_SET:, :]

train_x = data_train[:SIZE_OF_TRAINING_SET, 1:]
train_label = np.asarray(data_train[:SIZE_OF_TRAINING_SET, 0])


hl_sizes = [100, 100, 100, 100, 100]

network = NN.NeuralNetwork(training_data, validation_data, n_classes=5, n_hl=5, hl_sizes=hl_sizes, n_epochs=50)
network.train_neural_network()

network.prediction(data_test)
