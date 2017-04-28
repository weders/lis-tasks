import tensorflow as tf
import numpy as np




class NeuralNetwork:




    def __init__(self, training_data, validation_data, n_classes, n_hl, hl_sizes=None, activation_functions=None, batch_size=None , n_epochs=None):

        self.validation_data = validation_data
        self.training_data = training_data

        ##############################################################
        # Data Parsing
        ##############################################################

        self.x_train = np.asarray(self.training_data[:, 1:])
        self.x_validation = np.asarray(self.validation_data[:,1:])


        # one hot structure

        self.y_train = np.zeros((self.training_data.shape[0],n_classes))

        for i in range(0,self.y_train.shape[0]):
            self.y_train.itemset((i, int(self.training_data[i,0])), 1)

        self.y_validation = np.zeros((self.validation_data.shape[0], n_classes))

        for i in range(0, self.y_validation.shape[0]):
            self.y_validation.itemset((i, int(self.validation_data[i, 0])), 1)


        self.n_classes = n_classes
        self.n_hl = n_hl

        if batch_size is not None:
            self.batch_size = batch_size
        else:
            self.batch_size = 100


        #############################################################
        # Preprocessing
        #############################################################

        self.input_dimension = self.x_train.shape[1]

        #############################################################
        # Network Parameter Definition
        #############################################################

        self.activation_functions = activation_functions
        self.n_l = self.n_hl + 2
        self.n_nodes_layers = np.zeros((self.n_l,1))

        if hl_sizes == None:
            for i in range(0,self.n_l):

                if i == 0:
                    self.n_nodes_layers[i] = self.input_dimension

                if i is not 0 and i is not self.n_l-1:
                    self.n_nodes_layers[i] = 500

                if i == self.n_l - 1:
                    self.n_nodes_layers[i] = self.n_classes

        else:
            for i in range(0, self.n_l):

                if i == 0:
                    self.n_nodes_layers[i] = self.input_dimension

                if i is not 0 and i is not self.n_l - 1:
                    self.n_nodes_layers[i] = hl_sizes[i-1]

                if i == self.n_l - 1:
                    self.n_nodes_layers[i] = self.n_classes


        self.x_variable = tf.placeholder('float',[None,self.input_dimension])
        self.y_variable = tf.placeholder('float',[None,n_classes])

        #############################################################
        # Training Parameter Definition
        #############################################################

        if n_epochs == None:
            self.n_epochs = 10 # cycles of feedforward and backpropagation
        else:
            self.n_epochs = n_epochs


        #############################################################
        # Training Parameter Definition
        #############################################################

        self.LOG_DIR = 'logs/'




    def neural_network_model(self,data):

        layers = []
        for layer in range(1, self.n_l):
            layer_dict = {'weights': tf.Variable(tf.random_normal([int(self.n_nodes_layers[layer-1,0]), int(self.n_nodes_layers[layer,0])])),
                          'biases': tf.Variable(tf.random_normal([int(self.n_nodes_layers[layer,0])]))}
            layers.append(layer_dict)

        layer_computation = []

        for layer_number in range(0, self.n_l-1):

            if layer_number == 0:
                computation = tf.add(tf.matmul(data, layers[layer_number]['weights']), layers[layer_number]['biases'])
                if self.activation_functions == None or self.activation_functions[layer_number] == 'relu':
                    computation = tf.nn.relu(computation)
                if self.activation_functions[layer_number] == 'sigmoid':
                    computation = tf.nn.sigmoid(computation)
                if self.activation_functions[layer_number] == 'tanh':
                    computation = tf.nn.tanh(computation)
                if self.activation_functions[layer_number] == 'elu':
                    computation = tf.nn.elu(computation)
                if self.activation_functions[layer_number] == 'softplus':
                    computation = tf.nn.softplus(computation)
                if self.activation_functions[layer_number] == 'softsign':
                    computation = tf.nn.softsign(computation)




            if layer_number is not self.n_l - 1 and layer_number is not 0:
                computation = tf.add(tf.matmul(layer_computation[layer_number - 1], layers[layer_number]['weights']),
                                     layers[layer_number]['biases'])
                if self.activation_functions == None or self.activation_functions[layer_number] == 'relu':
                    computation = tf.nn.relu(computation)
                if self.activation_functions[layer_number] == 'sigmoid':
                    computation = tf.nn.sigmoid(computation)
                if self.activation_functions[layer_number] == 'tanh':
                    computation = tf.nn.tanh(computation)
                if self.activation_functions[layer_number] == 'elu':
                    computation = tf.nn.elu(computation)
                if self.activation_functions[layer_number] == 'softplus':
                    computation = tf.nn.softplus(computation)
                if self.activation_functions[layer_number] == 'softsign':
                    computation = tf.nn.softsign(computation)

            if layer_number == self.n_l - 2:
                computation = tf.add(tf.matmul(layer_computation[layer_number - 1], layers[layer_number]['weights']),
                                     layers[layer_number]['biases'])

            layer_computation.append(computation)

        output = layer_computation[-1]
        return output

    def train_neural_network(self):
        prediction = self.neural_network_model(self.x_variable)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=self.y_variable))

        # learning rate = 0.001
        optimizer = tf.train.AdamOptimizer().minimize(cost)



        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(0, self.n_epochs):
                epoch_loss = 0

                for i in range(0, int(self.x_train.shape[0]/self.batch_size)):
                    epoch_x = self.x_train[i * self.batch_size:(i + 1) * self.batch_size, :]
                    epoch_y = self.y_train[i * self.batch_size:(i + 1) * self.batch_size]

                    i, c = sess.run([optimizer, cost], feed_dict={self.x_variable: epoch_x, self.y_variable: epoch_y})
                    epoch_loss += c

                print('Epoch', epoch, 'completed out of', self.n_epochs, 'loss: ', epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y_variable, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            print('Accuracy:', accuracy.eval({self.x_variable: self.x_validation, self.y_variable: self.y_validation}))

            writer = tf.summary.FileWriter(self.LOG_DIR)
            writer.add_graph(sess.graph)

        return

    def prediction(self,prediction_data):
























