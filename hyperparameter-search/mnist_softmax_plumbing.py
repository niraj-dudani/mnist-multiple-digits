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

def train_and_test(data, learning_rate, r_lambda):
  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b
  
  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])
  
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
  
  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  
  # Train
  for _ in range(1000):
    batch_xs, batch_ys = data.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  
  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  
  accuracy_value = sess.run(
    accuracy,
    feed_dict = {
      x: data.test.images,
      y_: data.test.labels
    }
  )
  
  return accuracy_value



def main(_):
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  
  learning_rate = 0.5
  r_lambda = 1.0
  
  accuracy = train_and_test(mnist, learning_rate, r_lambda)
  print(accuracy)
  
  accuracy = train_and_test(mnist, 25, r_lambda)
  print(accuracy)



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='../data/train',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
