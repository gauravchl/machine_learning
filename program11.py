#!/usr/local/bin/python3

# Program to use frozen model

import tensorflow as tf
from PIL import Image
import numpy as np


# Using ssd_mobilenet_v1_coco
# http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz

FROZEN_MODEL_PATH = '/Users/om/models/mobilenet_v1_coco/frozen_inference_graph.pb'
IMAGE_PATH = '/Users/om/desktop/lemoncake.jpg'


# Function to load the frozen model
def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")
    return graph


# Helper function for obj detection model
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
    (im_height, im_width, 3)).astype(np.uint8)


print('\nLOADING MODEL')
detection_graph = load_graph(FROZEN_MODEL_PATH)
print('LOADING COMPLETE')



with tf.Session(graph=detection_graph) as sess:

    # Print all operations in any graph
    # for op in detection_graph.get_operations():
    #     print(op.name)


    image_tensor = detection_graph.get_tensor_by_name('prefix/image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('prefix/detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('prefix/detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('prefix/detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('prefix/num_detections:0')

    print('\nOPENING IMAGE')
    image = Image.open(IMAGE_PATH)
    image_np = load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis=0)
    print('IMAGE OPENED')

    print('\n PREDICTING RESULT')
    (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})


    print('\n RESULT: ')
    print('score: ', scores)
    print('detection classes: ', classes)
