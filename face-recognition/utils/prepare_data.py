#=======================
# Prepare data
#=======================

class PreparedData:
	def __init__(
		self,
		train_data,
		test_data,
		train_labels,
		test_labels
	):
		self.train_data = train_data
		self.test_data = test_data
		self.train_labels = train_labels
		self.test_labels = test_labels

def prepare_data(
		positive_file_path,
		negative_file_path,
		train_fraction
):
	import numpy as np
	
	positive = np.load(positive_file_path)
	negative = np.load(negative_file_path)
	
	n_positive = positive.shape[0]
	n_negative = negative.shape[0]
	
	positive_flattened = positive.reshape(n_positive, -1)
	negative_flattened = negative.reshape(n_negative, -1)
	
	positive_normalized = positive_flattened / positive_flattened.max() - 0.5
	negative_normalized = negative_flattened / negative_flattened.max() - 0.5
	
	train_positive_count = int(train_fraction * n_positive)
	train_positive = positive_normalized[:train_positive_count]
	test_positive = positive_normalized[train_positive_count:]
	train_positive_labels = np.zeros([train_positive.shape[0], 1])
	test_positive_labels = np.zeros([test_positive.shape[0], 1])
	
	train_negative_count = int(train_fraction * n_negative)
	train_negative = negative_normalized[:train_negative_count]
	test_negative = negative_normalized[train_negative_count:]
	train_negative_labels = np.ones([train_negative.shape[0], 1])
	test_negative_labels = np.ones([test_negative.shape[0], 1])
	
	train_data = np.vstack([train_positive, train_negative])
	test_data = np.vstack([test_positive, test_negative])
	
	train_labels = np.vstack([train_positive_labels, train_negative_labels])
	test_labels = np.vstack([test_positive_labels, test_negative_labels])
	
	prepared_data = PreparedData(
		train_data,
		test_data,
		train_labels,
		test_labels
	)
	
	return prepared_data
