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

def train_and_test(data, learning_rate, regularization_constant):
  # Create the model
  LAYER1_SIZE = 784 # input data dimension too
  LAYER2_SIZE = 10 # 128
  OUTPUT_SIZE = 10
  
  # input
  x = tf.placeholder(tf.float32, [None, LAYER1_SIZE])
  
  # layer 1
  W1 = tf.Variable(tf.zeros([LAYER1_SIZE, LAYER2_SIZE]))
  b1 = tf.Variable(tf.zeros([LAYER2_SIZE]))
  y1 = tf.sigmoid(tf.matmul(x, W1) + b1)
  
  # layer 2
  W2 = tf.Variable(tf.zeros([LAYER2_SIZE, OUTPUT_SIZE]))
  b2 = tf.Variable(tf.zeros([OUTPUT_SIZE]))
  
  # output
  # y = tf.sigmoid(tf.matmul(y1, W2) + b2)
  y = tf.matmul(y1, W2) + b2
  
  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])
  
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  
  # add regularization
  regularization_term = (
    tf.constant(regularization_constant) * (
      tf.reduce_sum(tf.square(W1)) +
      tf.reduce_sum(tf.square(W2))
    )
  )
  
  loss = cross_entropy + regularization_term
  train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
  
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


class SearchResult:
  def __init__(self, hyperparameters, accuracy):
    self.hyperparameters = hyperparameters
    self.accuracy = accuracy


def search(data, hyperparameter_candidates):
  import numpy as np
  
  accuracies = []
  
  for hyperparameters in hyperparameter_candidates:
    learning_rate, regularization_constant = hyperparameters
    
    msg = "Learning rate = {} ; Regularization parameter = {} ; "
    msg = msg.format(learning_rate, regularization_constant)
    
    print(msg, flush = True, end = '')
    accuracy = train_and_test(data, learning_rate, regularization_constant)
    print(accuracy, flush = True)
    
    accuracies.append(accuracy)
  
  argmax = np.argmax(accuracies)
  
  best_accuracy = accuracies[argmax]
  best_hyperparameters = hyperparameter_candidates[argmax]
  
  search_result = SearchResult(
    best_hyperparameters,
    best_accuracy
  )
  
  return search_result



def main(_):
  import grid_search
  
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  
  grid_size = 20
  n_samples = 30
  shrink_factor = 10
  n_search_levels = 1
  
  # Parameter range
  ln_min = -3
  ln_max = 1
  
  rg_min = -3
  rg_max = 1
  
  for i_search_level in range(n_search_levels):
    print('[Search Level {}]'.format(i_search_level))
    print('Learning rate range: [{}, {}]'.format(ln_min, ln_max))
    print('Regularization constant range: [{}, {}]'.format(rg_min, rg_max))
    
    hyperparameter_candidates = [
      grid_search.grid_search(
        (ln_min, rg_min),
        (ln_max, rg_max),
        grid_size
      )
      for _ in range(n_samples)
    ]
    
    
    print(hyperparameter_candidates)
    
    #~ hyperparameter_candidates = [
      #~ (0.5, 0.0),
      #~ (0.5, 0.1),
      #~ (0.5, 1.0),
      #~ (0.5, 10.0),
    #~ ]
    
    #~ search_result = search(mnist, hyperparameter_candidates)
    
    #~ print(search_result.accuracy, search_result.hyperparameters)
  
  pass


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='../data/train',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
