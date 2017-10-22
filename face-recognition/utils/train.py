#=======================
# Train and predict
#=======================

def train_and_predict(model, data, n_iterations, logging_frequency = 10):
	import tensorflow as tf
	
	loss_history = []
	
	with tf.Session() as session:
		session.run(tf.global_variables_initializer())
		
		for i in range(n_iterations):
			loss_value, _ = session.run(
				[model.loss, model.train_op],
				feed_dict = {
					model.train_x: data.train_data,
					model.train_y: data.train_labels
				}
			)
			
			if i % logging_frequency == 0:
				print("Loss after {} iterations: {}".format(i, loss_value))
			
			loss_history.append(loss_value)
		
		y_ = session.run(
			model.z, 
			feed_dict = {
				model.train_x: data.test_data
			}
		)
		y_ = y_ > 0.5
		
		weights_trained = session.run(model.weights)
	
	is_correct = (y_ == data.test_labels)
	n_correct = sum(is_correct)[0]
	
	n_test = y_.shape[0]
	
	accuracy = n_correct / n_test
	
	return weights_trained, accuracy
