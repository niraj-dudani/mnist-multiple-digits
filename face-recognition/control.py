import utils

#=======================
# Globals
#=======================
positive_file_path = 'data/faces.npy'
negative_file_path = 'data/nonfaces.npy'
train_fraction = 0.8

learning_rate = 0.05
n_iterations = 1000

weights_init_standard_deviation = 1e-3
bias_init = 0.1

logging_frequency = 10

data = utils.prepare_data(
	positive_file_path,
	negative_file_path,
	train_fraction
)

n_input_neurons = data.train_data.shape[1]

def random_weights(n_weights, stddev):
	import tensorflow as tf
	
	weights_init = tf.random_normal(
		[n_weights, 1],
		stddev = stddev,
		name = "weights_init"
	)
	
	return weights_init

weights_init_random = random_weights(
	n_input_neurons,
	weights_init_standard_deviation
)

model = utils.build(
	learning_rate,
	weights_init_random,
	bias_init
)

weights, accuracy = utils.train_and_predict(
	model,
	data,
	n_iterations,
	logging_frequency
)

print("Accuracy: {}%".format(accuracy * 100))
