import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn import preprocessing


#################################################
# load train data and test data
train = pd.read_hdf("train.h5", "train")
final = pd.read_hdf("test.h5", "test")
features = list(train)[1:]
data_final = final[features]

#################################################
# split into train and validate data
size_of_train = 42000
# train data
data_train = train[0:size_of_train]
targets_train = data_train['y']
features_train = data_train[features]
# validate data
data_test = train[size_of_train:]
targets_test = data_test['y']
targets_test = pd.get_dummies(targets_test)
features_test = data_test[features]
features_test = preprocessing.scale(features_test)


##################################################
# tensorflow
# defining our model
n_nodes_hl1 = 100
n_nodes_hl2 = 100
n_nodes_hl3 = 100
n_nodes_hl4 = 100
n_nodes_hl5 = 100

n_classes = 5
batch_size = 60
n_batches = int(features_train.shape[0] / batch_size)

x = tf.placeholder('float', [None, 100])
y = tf.placeholder('float')

# def neuralnetwork


def neural_network_model(data):

    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([100, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
    hidden_4_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl4])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl4]))}
    hidden_5_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl4, n_nodes_hl5])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl5]))}
    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl5, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))}

    # inputs_Data * weights + biases
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.sigmoid(l3)

    l4 = tf.add(tf.matmul(l3, hidden_4_layer['weights']), hidden_4_layer['biases'])
    l4 = tf.nn.sigmoid(l4)

    l5 = tf.add(tf.matmul(l4, hidden_5_layer['weights']), hidden_5_layer['biases'])
    l5 = tf.nn.sigmoid(l5)

    output = tf.matmul(l5, output_layer['weights']) + output_layer['biases']
    return output


# define training neural network
def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    hm_epochs = 100
    # start session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # training neural network
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(n_batches):
                data_batch = data_train[_ * batch_size:_ * batch_size + batch_size]
                epoch_x = data_batch[features]
                epoch_x = preprocessing.scale(epoch_x)
                epoch_y = data_batch['y']
                epoch_y = pd.get_dummies(epoch_y)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy', accuracy.eval({x: features_test, y: targets_test}))

        # predict accuracy on test data
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy', accuracy.eval({x: features_test, y: targets_test}))
        pred_ = sess.run(prediction, feed_dict={x: data_final})
        decoded = sess.run(tf.argmax(pred_, axis=1))
        decoded = np.array(decoded).reshape((-1, 1))
        data_final['y'] = decoded
        result = data_final['y']
        result.to_csv('result.csv', header=['y'])


train_neural_network(x)
