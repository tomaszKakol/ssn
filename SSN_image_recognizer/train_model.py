from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from cnn_model_fn import cnn_model_fn

import os
import struct
from struct import unpack
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.contrib import learn

tf.logging.set_verbosity(tf.logging.INFO)


def unpack_drawing(file_handle):
    key_id, = unpack('Q', file_handle.read(8))
    countrycode, = unpack('2s', file_handle.read(2))
    recognized, = unpack('b', file_handle.read(1))
    timestamp, = unpack('I', file_handle.read(4))
    n_strokes, = unpack('H', file_handle.read(2))
    image = []
    for i in range(n_strokes):
        n_points, = unpack('H', file_handle.read(2))
        fmt = str(n_points) + 'B'
        x = unpack(fmt, file_handle.read(n_points))
        y = unpack(fmt, file_handle.read(n_points))
        image.append((x, y))

    return {
        'key_id': key_id,
        'countrycode': countrycode,
        'recognized': recognized,
        'timestamp': timestamp,
        'image': image
    }


def unpack_drawings(filename):
    with open(filename, 'rb') as f:
        while True:
            try:
                yield unpack_drawing(f)
            except struct.error:
                break


def throwout_wrong_images(binary_file, all_drawings):
    indexes_recognised = []
    drawings_recognised = []
    for i,drawing in enumerate(unpack_drawings(binary_file)):
        if ((drawing['recognized']) ==1):
            indexes_recognised.append(i)
    for inx in indexes_recognised:
        drawings_recognised.append(all_drawings[inx])
    drawings_recognised_np = np.asarray(drawings_recognised)
    return drawings_recognised_np



def main(unused_argv):

    T = np.load('triangles.npy')
    C = np.load('circles.npy')
    S = np.load('squares.npy')
    H = np.load('hexagones.npy')

    triangles_all = T.reshape((T.shape[0], 28, 28))
    circles_all = C.reshape((C.shape[0], 28, 28))
    squares_all = S.reshape((S.shape[0], 28, 28))
    hexagons_all = H.reshape((H.shape[0], 28, 28))

    # prepare datasets without inapropriate images
    triangles1 = []
    circles1 = []
    squares1 = []
    hexagons1 = []

    triangles1 = np.float32(throwout_wrong_images('triangle.bin', triangles_all))
    circles1 = np.float32(throwout_wrong_images('circle.bin', circles_all))
    squares1 = np.float32(throwout_wrong_images('square.bin', squares_all))
    hexagons1 = np.float32(throwout_wrong_images('hexagon.bin', hexagons_all))

    triangles = np.float32(triangles1[:10000])
    circles = np.float32(circles1[:10000])
    squares = np.float32(squares1[:10000])
    hexagons = np.float32(hexagons1[:10000])

    number_of_samples = [triangles.shape[0], circles.shape[0], squares.shape[0], hexagons.shape[0]]
    number_of_samples_fibb = [triangles.shape[0], triangles.shape[0] + circles.shape[0],
                            triangles.shape[0] + circles.shape[0] + squares.shape[0], sum(number_of_samples)]

    # creates input and output matrix
    X = np.concatenate((triangles, circles, squares, hexagons), axis=0)
    Y = np.zeros(X.shape[0])
    Y[:number_of_samples_fibb[0]] = np.int(0)
    Y[number_of_samples_fibb[0]:number_of_samples_fibb[1]] = np.int(1)
    Y[number_of_samples_fibb[1]:number_of_samples_fibb[2]] = np.int(2)
    Y[number_of_samples_fibb[2]:number_of_samples_fibb[3]] = np.int(3)

    # split data
    train_data, eval_data, train_labels, eval_labels = train_test_split(X, Y, test_size=0.2, random_state=42)


    # Create the Estimator and save to the file
    dirname = os.path.dirname(os.path.abspath(__file__))
    shape_classifier = learn.Estimator(
      model_fn=cnn_model_fn, model_dir=os.path.join(dirname, 'net', 'shape_model'))

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    shape_classifier.fit(
      x=train_data,
      y=train_labels,
      batch_size=128,
      steps=15000,
      monitors=[logging_hook])

    # Configure the accuracy metric for evaluation
    metrics = {
      "accuracy":
          learn.MetricSpec(
              metric_fn=tf.metrics.accuracy, prediction_key="classes"),
    }

    # Evaluate the model and print results
    eval_results = shape_classifier.evaluate(
      x=eval_data, y=eval_labels, metrics=metrics)
    print(eval_results)


if __name__ == "__main__":
  tf.app.run()
