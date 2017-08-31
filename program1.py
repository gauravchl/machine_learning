#!/usr/local/bin/python3

# Print hello world
import tensorflow as tf
sess = tf.Session()

node = tf.constant('Hello world!');
print(sess.run(node))
