#=======================
# Train and predict
#=======================

def train(
		session,
		model,
		data,
		n_iterations,
		logging_frequency = 10
):
	import tensorflow as tf
	
	loss_history = []
	
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
	
	weights_trained = session.run(model.weights)
	
	return weights_trained, loss_history
