def resize(images):
	import skimage
	
	resized = [
		skimage.transfom.resize(image, [24, 240])
		for image in images
	]
	
	return resized
