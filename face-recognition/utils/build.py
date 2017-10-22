#=======================
# Build
#=======================

class Model:
	pass

def build(
	learning_rate,
	weights_init,
	bias_init = 0.1
):
	import numpy as np
	import tensorflow as tf
	
	train_x = tf.placeholder(name = "train_data", dtype = np.float32)
	train_y = tf.placeholder(name = "train_label", dtype = np.float32)
	
	weights = tf.Variable(weights_init, name = "weights")
	
	bias = tf.Variable(bias_init, name = "bias")
	
	h = tf.matmul(train_x, weights) + bias
	
	z = tf.sigmoid(h)
	
	z_plus = z + 1e-6
	
	loss = -tf.reduce_mean(
		train_y * tf.log(z_plus) +
		(1 - train_y) * tf.log(1 - z_plus)
	)
	
	dw, db = tf.gradients(loss, [weights, bias])
	
	weights_update = tf.assign_add(weights, -learning_rate * dw)
	bias_update = tf.assign_add(bias, -learning_rate * db)
	
	with tf.control_dependencies([weights_update, bias_update]):
		train_op = tf.no_op()
	
	model = Model()
	model.train_x = train_x
	model.train_y = train_y
	model.weights = weights
	model.h = h
	model.z = z
	model.loss = loss
	model.train_op = train_op
	
	return model
