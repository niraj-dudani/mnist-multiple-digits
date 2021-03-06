# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  LAYER1_SIZE = 784 # input data dimension too
  LAYER2_SIZE = 2 # 128
  OUTPUT_LAYER_SIZE = 10
  LAMBDA = 1 # regularization factor for the weights

  # input
  x = tf.placeholder(tf.float32, [None, LAYER1_SIZE])

  # layer 1
  W1 = tf.Variable(tf.zeros([LAYER1_SIZE, LAYER2_SIZE]))
  b1 = tf.Variable(tf.zeros([LAYER2_SIZE]))
  y1 = tf.sigmoid(tf.matmul(x, W1) + b1)

  # layer 2
  W2 = tf.Variable(tf.zeros([LAYER2_SIZE, OUTPUT_LAYER_SIZE]))
  b2 = tf.Variable(tf.zeros([OUTPUT_LAYER_SIZE]))

  # output
  y = tf.matmul(y1, W2) + b2

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, OUTPUT_LAYER_SIZE])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

  # add regularization
  regularization_term = tf.constant(LAMBDA) * (tf.reduce_sum(tf.square(W1)) + tf.reduce_sum(tf.square(W2)))

  loss = cross_entropy + regularization_term
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='../data/train',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
