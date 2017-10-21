def digits(wild_image):
	import crop
	import chop
	import numpy as np
	
	cropped = crop.crop(wild_image)
	chopped = chop.chop(cropped)
	
	n_images = chopped.shape[0]
	
	flattened = np.reshape(n_images, -1)
	
	return flattened
