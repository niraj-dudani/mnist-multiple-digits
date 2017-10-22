#=======================
# Globals
#=======================
faces_file = 'data/faces.npy'
non_faces_file = 'data/nonfaces.npy'
train_fraction = 0.8

learning_rate = 0.05
n_iterations = 1000

weights_init_standard_deviation = 1e-3
bias_init = 0.1

logging_frequency = 10


#=======================
# Prepare data
#=======================
import numpy as np

faces = np.load(faces_file)
non_faces = np.load(non_faces_file)

n_faces = faces.shape[0]
n_non_faces = non_faces.shape[0]

faces_flattened = faces.reshape(n_faces, -1)
non_faces_flattened = non_faces.reshape(n_non_faces, -1)

faces_normalized = faces_flattened / faces_flattened.max() - 0.5
non_faces_normalized = non_faces_flattened / non_faces_flattened.max() - 0.5

train_faces_count = int(train_fraction * n_faces)
train_faces = faces_normalized[:train_faces_count]
test_faces = faces_normalized[train_faces_count:]
train_faces_labels = np.zeros([train_faces.shape[0], 1])
test_faces_labels = np.zeros([test_faces.shape[0], 1])

train_non_faces_count = int(train_fraction * n_non_faces)
train_non_faces = non_faces_normalized[:train_non_faces_count]
test_non_faces = non_faces_normalized[train_non_faces_count:]
train_non_faces_labels = np.ones([train_non_faces.shape[0], 1])
test_non_faces_labels = np.ones([test_non_faces.shape[0], 1])

train_data = np.vstack([train_faces, train_non_faces])
test_data = np.vstack([test_faces, test_non_faces])

train_labels = np.vstack([train_faces_labels, train_non_faces_labels])
test_labels = np.vstack([test_faces_labels, test_non_faces_labels])


#=======================
# Build
#=======================
import tensorflow as tf

n_dimensions = train_data.shape[1]

train_x = tf.placeholder(name = "train_data", dtype = np.float32)
train_y = tf.placeholder(name = "train_label", dtype = np.float32)

weights_init = tf.random_normal(
	[n_dimensions, 1],
	stddev = weights_init_standard_deviation,
	name = "weights_init"
)

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

#=======================
# Train + Predict
#=======================
loss_history = []

with tf.Session() as session:
	session.run(tf.global_variables_initializer())
	
	for i in range(n_iterations):
		loss_value, _ = session.run(
			[loss, train_op],
			feed_dict = {
				train_x: train_data,
				train_y: train_labels
			}
		)
		
		if i % logging_frequency == 0:
			print("Loss after {} iterations: {}".format(i, loss_value))
		
		loss_history.append(loss_value)
	
	y_ = session.run(z, feed_dict = {train_x: test_data})
	y_ = y_ > 0.5
	
	weights_trained = session.run(weights)

is_correct = (y_ == test_labels)
n_correct = sum(is_correct)[0]

n_test = y_.shape[0]

accuracy = n_correct / n_test

print("Accuracy: {}%".format(accuracy * 100))

#=======================
# Viz
#=======================

import matplotlib.pyplot as plt

plt.subplot(2,1,1)
plt.imshow(faces_normalized, 'gray')
plt.subplot(2,1,2)
plt.imshow(non_faces_normalized, 'gray')

plt.figure()
n_pixels = faces_flattened.shape[1]
indices = list(range(n_pixels))
faces_avg = faces_normalized.mean(axis = 0)
non_faces_avg = non_faces_normalized.mean(axis = 0)
plt.plot(indices, faces_avg)
plt.plot(indices, non_faces_avg)

plt.figure()
plt.plot(loss_history)

plt.show()
