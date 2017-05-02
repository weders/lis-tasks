import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
import tensorflow as tf


# load train data and test data
train = pd.read_hdf("data/train.h5", "train")
final = pd.read_hdf("data/test.h5", "test")
features = list(train)[1:]
data_final = final[features]
data_final = data_final.as_matrix()

#################################################
# split into train and validate data
size_of_train = 40000
# train data
data_train = train[0:size_of_train]
targets_train = data_train['y']
features_train = data_train[features]
features_train = features_train.as_matrix()
targets_train = pd.get_dummies(targets_train)
targets_train = targets_train.as_matrix()

# validate data
data_test = train[size_of_train:]
targets_test = data_test['y']
features_test = data_test[features]
features_test = features_test.as_matrix()
targets_test = pd.get_dummies(targets_test)
targets_test = targets_test.as_matrix()
# create a model

model = Sequential()

# add layers
model.add(Dense(units=1500, input_dim=100))
model.add(Activation('relu'))
model.add(Dense(units=1000))
model.add(Activation('relu'))
model.add(Dense(units=500))
model.add(Activation('relu'))
model.add(Dense(units=5))
model.add(Activation('softmax'))

# compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# fit data
model.fit(features_train, targets_train, epochs=40, batch_size=500)

# evaluate on validate(test) set
loss_and_metrics = model.evaluate(features_test, targets_test, batch_size=500)

# predict
classes = model.predict(data_final, batch_size=500)

with tf.Session() as sess:
    decoded = sess.run(tf.argmax(classes, axis=1))
    decoded = np.array(decoded).reshape((-1, 1))
    print(decoded)
    final['y'] = decoded
    result = final['y']
    result.to_csv('result.csv', header=['y'])
