import tensorflow as tf 
import pandas as pd 
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from scipy.special import expit


        

#read train data in panda data frame
data_set = pd.read_csv("./train.csv" , index_col = 0  )
data_train = data_set[:800]
data_test = data_set[800:]
#extract features and labels
features_columns = list(data_set)[1:]
features_train = data_train[features_columns]
features_test = data_test[features_columns]
targets_train = pd.get_dummies(data_train['y'])
targets_test = pd.get_dummies(data_test['y'])
#read test_data --> final 
data_final = pd.read_csv("./test.csv" , index_col = 0  )

#drop correlated columns
features_test=features_test.drop(['x8','x9','x14'], axis=1)
features_train=features_train.drop(['x8','x9','x14'], axis=1)
data_final=data_final.drop(['x8','x9','x14'], axis=1)






#defining our model
n_nodes_hl1 = 150
n_nodes_hl2 = 150
#n_nodes_hl3 = 1000

n_classes = 3
batch_size = 60
n_batches = int(features_train.shape[0]/batch_size)



x = tf.placeholder('float',[None,12])
y = tf.placeholder('float')

def neural_network_model(data):

        hidden_1_layer = {'weights' : tf.Variable(tf.random_normal([12, n_nodes_hl1])),
                          'biases' : tf.Variable(tf.random_normal([n_nodes_hl1]))}
        hidden_2_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                          'biases' : tf.Variable(tf.random_normal([n_nodes_hl2]))}
        #hidden_3_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
        #                  'biases' : tf.Variable(tf.random_normal([n_nodes_hl3]))}
        output_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                          'biases' : tf.Variable(tf.random_normal([n_classes]))}

        #inputs_Data * weights + biases
        l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
        l1 = tf.nn.relu(l1)

        l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
        l2 = tf.nn.relu(l2)

        #l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
        #l3 = tf.nn.relu(l3)

        output = tf.matmul(l2,output_layer['weights'])+ output_layer['biases']

        return output

# define training neural network
def train_neural_network(x):
        prediction = neural_network_model(x)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels = y))
        optimizer= tf.train.AdamOptimizer().minimize(cost)
        hm_epochs = 500
        #start session
        with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                #training neural network
                for epoch in range(hm_epochs):
                        epoch_loss = 0
                        for _ in range(n_batches):
                                data_batch = data_train[_*batch_size:_*batch_size+batch_size]
                                epoch_x = data_batch[features_columns]
                                epoch_x = epoch_x.drop(['x8','x9','x14'],axis = 1)
                                epoch_y = data_batch['y']
                                epoch_y = pd.get_dummies(epoch_y)
                                _ , c = sess.run([optimizer,cost], feed_dict= {x:epoch_x,y:epoch_y})
                                epoch_loss += c
                        print('Epoch', epoch, 'completed out of' ,hm_epochs ,'loss:' , epoch_loss)
                        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
                        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
                        print('Accuracy' , accuracy.eval({x:features_test,y:targets_test}))

                #predict accuracy on test data
                correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
                accuracy = tf.reduce_mean(tf.cast(correct,'float'))
                print('Accuracy' , accuracy.eval({x:features_test,y:targets_test}))
                pred_ = sess.run(prediction, feed_dict={x:data_final})
                decoded = sess.run(tf.argmax(pred_, axis=1))
                decoded = np.array(decoded).reshape((-1,1))
                data_final['y'] = decoded
                result = data_final['y']
                result.to_csv('result.csv',header=['y'])
                



train_neural_network(x)





