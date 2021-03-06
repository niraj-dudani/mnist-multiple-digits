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

def train_and_test(data, learning_rate, regularization_constant, iterations, batch_size, n_hidden):
  # Create the model
  LAYER1_SIZE = 784 # input data dimension too
  LAYER2_SIZE = int(n_hidden)
  OUTPUT_SIZE = 10
  
  # input
  x = tf.placeholder(tf.float32, [None, LAYER1_SIZE])
  
  # layer 1
  W1 = tf.Variable(tf.random_normal([LAYER1_SIZE, LAYER2_SIZE]))
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
  for _ in range(int(iterations)):
    batch_xs, batch_ys = data.train.next_batch(int(batch_size))
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
  
  sess.close()
  
  return accuracy_value


class SearchResult:
  def __init__(self, hyperparameters, accuracy):
    self.hyperparameters = hyperparameters
    self.accuracy = accuracy


def search(data, hyperparameter_candidates):
  import numpy as np
  
  accuracies = []
  
  for hyperparameters in hyperparameter_candidates:
    learning_rate, regularization_constant, iterations, batch_size, hidden_neurons = hyperparameters
    
    print()
    print()
    print("Learning rate =", learning_rate)
    print("Regularization constant =", regularization_constant)
    print("Iterations =", iterations)
    print("Batch size =", batch_size)
    print("Hidden neurons =", hidden_neurons)
    
    accuracy = train_and_test(
      data,
      learning_rate,
      regularization_constant,
      iterations,
      batch_size,
      hidden_neurons
    )
    print("Accuracy =", accuracy)
    
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
  import math
  
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  
  grid_size = 40
  n_samples = 20
  zoom_factor = 5
  n_search_levels = 2
  
  # Parameter range
  ln_min = -3
  ln_max = 0
  
  rg_min = -7
  rg_max = -3
  
  it_min = 3
  it_max = 5
  
  bt_min = 1
  bt_max = 3
  
  hd_min = 1.3
  hd_max = 2.8
  
  for i_search_level in range(n_search_levels):
    print()
    print('[[ Search Level {} ]]'.format(i_search_level))
    print('Learning rate range: [{}, {}]'.format(ln_min, ln_max))
    print('Regularization constant range: [{}, {}]'.format(rg_min, rg_max))
    print('Iteration range: [{}, {}]'.format(it_min, it_max))
    print('Batch size range: [{}, {}]'.format(bt_min, bt_max))
    print('Hidden neurons range: [{}, {}]'.format(hd_min, hd_max))
    
    hyperparameter_candidates = grid_search.grid_search(
      (ln_min, rg_min, it_min, bt_min, hd_min),
      (ln_max, rg_max, it_max, bt_max, hd_max),
      grid_size,
      n_samples
    )
    
    
    print(hyperparameter_candidates)
    
    #~ hyperparameter_candidates = [
      #~ (0.5, 0.0),
      #~ (0.5, 0.1),
      #~ (0.5, 1.0),
      #~ (0.5, 10.0),
    #~ ]
    
    #~ import pdb ; pdb.set_trace()
    
    search_result = search(mnist, hyperparameter_candidates)
    
    
    print("Search done.")
    print("Accuracy:", search_result.accuracy)
    print("Hyperparameters:", search_result.hyperparameters)
    
    #~ search_result = SearchResult(
      #~ hyperparameter_candidates[3],
      #~ 0.9
    #~ )
    
    ln_best, rg_best, it_best, bt_best, hd_best = search_result.hyperparameters
    
    ln_best_exponent = math.log10(ln_best)
    rg_best_exponent = math.log10(rg_best)
    it_best_exponent = math.log10(it_best)
    bt_best_exponent = math.log10(bt_best)
    hd_best_exponent = math.log10(hd_best)
    
    range_length_current_ln = ln_max - ln_min
    range_length_current_rg = rg_max - rg_min
    range_length_current_it = it_max - it_min
    range_length_current_bt = bt_max - bt_min
    range_length_current_hd = hd_max - hd_min
    
    range_length_new_ln = range_length_current_ln / zoom_factor
    range_length_new_rg = range_length_current_rg / zoom_factor
    range_length_new_it = range_length_current_it / zoom_factor
    range_length_new_bt = range_length_current_bt / zoom_factor
    range_length_new_hd = range_length_current_hd / zoom_factor
    
    ln_min = ln_best_exponent - range_length_new_ln / 2
    ln_max = ln_best_exponent + range_length_new_ln / 2
    
    rg_min = rg_best_exponent - range_length_new_rg / 2
    rg_max = rg_best_exponent + range_length_new_rg / 2
    
    it_min = it_best_exponent - range_length_new_it / 2
    it_max = it_best_exponent + range_length_new_it / 2
    
    bt_min = bt_best_exponent - range_length_new_bt / 2
    bt_max = bt_best_exponent + range_length_new_bt / 2
    
    hd_min = hd_best_exponent - range_length_new_hd / 2
    hd_max = hd_best_exponent + range_length_new_hd / 2


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--data_dir',
    type=str,
    default='../data/train',
    help='Directory for storing input data'
  )
  
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
