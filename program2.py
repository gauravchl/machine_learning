#!/usr/local/bin/python3

# constant nodes, and operations on it
import tensorflow as tf
sess = tf.Session()

node1 = tf.constant(4);
node2 = tf.constant(5);

node3 = node1 + node2
node4 = tf.add(node1, node2)

print(sess.run(node3))
print(sess.run(node4))
