#=======================
# Predict
#=======================

def predict(session, model, data):
	import tensorflow as tf
	
	y_ = session.run(
		model.z, 
		feed_dict = {
			model.train_x: data.test_data
		}
	)
	y_ = y_ > 0.5
	
	is_correct = (y_ == data.test_labels)
	n_correct = sum(is_correct)[0]
	
	n_test = y_.shape[0]
	
	accuracy = n_correct / n_test
	
	return accuracy
