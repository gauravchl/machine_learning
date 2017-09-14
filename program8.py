#!/usr/local/bin/python3

# Estimator
# https://www.tensorflow.org/get_started/get_started#tfestimator

# Feature columns - pass into estimator when initialize
# Training data(x_train, y_train) - return from input fun
# Input function (input_fun) - pass into estimator when training or evaluating estimator


# Dont log all warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Actual code starts here
import tensorflow as tf
import numpy as np

print('\n# Program8 Estimators\n')

feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]
estimator = tf.estimator.LinearRegressor(feature_columns = feature_columns)

x_train = np.array([1., 2., 3., 4.])
y_train = np.array([1., 2., 3., 4.])

# functon returns x and y
input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train},
    y_train,
    batch_size = 4,
    num_epochs = None,
    shuffle = True,
)


train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train},
    y_train,
    batch_size = 4,
    num_epochs = 400,
    shuffle = False,
)

estimator.train(input_fn=input_fn, steps=100)

train_metrics = estimator.evaluate(input_fn=train_input_fn)

print(train_metrics)

# Finally call estimator.predict https://www.tensorflow.org/api_docs/python/tf/estimator/LinearRegressor
