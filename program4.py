#!/usr/local/bin/python3

# linear model using variables
import tensorflow as tf
sess = tf.Session()

W = tf.Variable(.3);
b = tf.Variable(-.3);
x = tf.placeholder(tf.float32)

linear_model = W * x + b

init = tf.global_variables_initializer()
sess.run(init) # initialize the variable
print(sess.run(linear_model, {x: [1,2,3,4,5]}))
