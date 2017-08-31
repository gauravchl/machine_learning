#!/usr/local/bin/python3

# placeholder in tensorflow
import tensorflow as tf
sess = tf.Session()

node1 = tf.placeholder(tf.int32);
node2 = tf.placeholder(tf.int32);

node3 = node1 + node2
print(sess.run(node3, {node1: 4, node2: 6}))
