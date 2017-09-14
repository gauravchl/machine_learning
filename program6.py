#!/usr/local/bin/python3

# Optimizer

import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Training data
x = tf.placeholder(tf.float32) # Input
y = tf.placeholder(tf.float32) # Output

W = tf.Variable(1.)
b = tf.Variable(1.)

linear_model = W * x + b
square_deltas = tf.square( linear_model - y )
cost = tf.reduce_sum(square_deltas)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# print(sess.run(square_deltas, { x: [1,2, 3, 4, 5], y: [1,2, 3, 4, 5]}))


optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(cost)
for i in range(200):
  print(sess.run([W, b, cost], {x: [1,2,3,4,5], y: [1,2,3,4,5]}))
  sess.run(train, {x: [1,2,3,4,5], y: [1,2,3,4,5]})
