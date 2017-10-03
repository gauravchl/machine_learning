#!/usr/local/bin/python3

# Save/Export model
# https://www.tensorflow.org/serving/serving_basic

# Dont log all warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

print('\n# Program9 Export model\n')

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


optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(cost)

print('Trainging model...')
for i in range(200):
  print(sess.run([W, b, cost], {x: [1,2,3,4,5], y: [1,2,3,4,5]}))
  sess.run(train, {x: [1,2,3,4,5], y: [1,2,3,4,5]})


print('Training done')
print(sess.run([W, b, linear_model], {x: [1,2,3,4,5]}))



print('Exporting...')
# Exporting model starts

export_path_base = 'models/'
path = 'models/'
version = '1.0.1'

FLAGS = tf.app.flags.FLAGS
export_path = os.path.join( tf.compat.as_bytes(path), tf.compat.as_bytes(version))
print('Exporting trained model to', export_path)

builder = tf.saved_model.builder.SavedModelBuilder(export_path)

tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
tensor_info_y = tf.saved_model.utils.build_tensor_info(y)
prediction_signature = (
  tf.saved_model.signature_def_utils.build_signature_def(
      inputs={'x': tensor_info_x},
      outputs={'scores': tensor_info_y},
      method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')


builder.add_meta_graph_and_variables(
    sess, [tf.saved_model.tag_constants.SERVING],
    signature_def_map={
        'predict_output':
           prediction_signature,
    },
    legacy_init_op=legacy_init_op)

builder.save()

print('Model exported')
