#!/usr/local/bin/python3

# Cost/Loss function

import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Training data
x = tf.placeholder(tf.float32) # Input
y = tf.placeholder(tf.float32) # Output

W = tf.Variable(1.)
b = tf.Variable(2.)

linear_model = W * x + b
square_deltas = tf.square( linear_model - y )
cost = tf.reduce_sum(square_deltas)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# print(sess.run(square_deltas, { x: [1,2, 3, 4, 5], y: [1,2, 3, 4, 5]}))

print('Cost(1.0, 2.0) = %s'%sess.run(cost, { x: [1,2, 3, 4, 5], y: [1,2, 3, 4, 5]}))
sess.run([tf.assign(W, 2), tf.assign(b, 4)])
print('Cost(2.0, 4.0) = %s'%sess.run(cost, { x: [1,2, 3, 4, 5], y: [1,2, 3, 4, 5]}))

sess.run([tf.assign(W, 1), tf.assign(b, 4)])
print('Cost(1.0, 4.0) = %s'%sess.run(cost, { x: [1,2, 3, 4, 5], y: [1,2, 3, 4, 5]}))

sess.run([tf.assign(W, 1), tf.assign(b, 0)])
print('Cost(1.0, 0.0) = %s'%sess.run(cost, { x: [1,2, 3, 4, 5], y: [1,2, 3, 4, 5]}))
