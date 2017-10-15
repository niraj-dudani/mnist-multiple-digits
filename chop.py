def chop(cropped_image):
	import numpy as np
	
	imgparts = np.split(cropped_image, 10, 1)
	
	return imgparts
