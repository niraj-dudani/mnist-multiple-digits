def resize(images):
	import skimage
	
	resized = [
		skimage.transfom.resize(image, [240, 24])
		for image in images
	]
	
	return resized
