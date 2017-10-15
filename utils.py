def display_image(
		image,
		title = None,
		cmap = None,
		subplot = (1, 1, 1),
		facecolor = (0.7, 0.7, 0.7)
):
	from matplotlib import pyplot as plt
	
	figure = plt.figure(title)
	
	rect = figure.patch
	rect.set_facecolor(facecolor)
	
	plt.subplot(*subplot)
	plt.imshow(image, cmap = cmap)
	plt.axis('off')

def show_images():
	from matplotlib import pyplot as plt
	plt.show()

def chop(cropped_image):
	import numpy as np
	
	imgparts = np.split(cropped_image, 10, 1)
	
	n_cols = imgparts[0].shape[1]
	
	chopped_array = np.concatenate(imgparts)
	chopped_array = chopped_array.reshape(10, -1, n_cols)
	
	return chopped_array

def resize(image):
	from skimage import transform
	
	resized = transform.resize(image, [28, 280], mode = 'reflect')
	
	return resized

def crop(raw_image_path):
	import numpy as np
	import skimage as ski
	from skimage import io
	
	I = ski.io.imread(raw_image_path)
	
	# help(ski.transform.rescale) # DO THIS LATER. RESIZE INPUT IMAGE SO IT HAS UPPER BOUNDS ON SIZE
	
	I_gray = ski.color.rgb2gray(I)
	
	# The thresholding methods I used below were borrowed/adapted from here http://scikit-image.org/docs/dev/auto_examples/xx_applications/plot_thresholding.html
	# from skimage.filters import threshold_local
	# local_block_size = 101
	# I_ = ski.filters.threshold_otsu(I_gray)
	# # I_ = ski.filters.threshold_local(I_gray, local_block_size, offset=50)
	# I_thresh = I_gray > I_
	
	# from skimage.morphology import disk
	# from skimage.filters import threshold_otsu, rank
	# from skimage.util import img_as_ubyte
	
	# radius = 300
	# selem = disk(radius)
	
	# I_gray_ = ski.img_as_ubyte(I_gray)
	# I_ = rank.otsu(I_gray_, selem) # local otsu
	# # I_ = threshold_otsu(I_gray_)
	# # global_otsu = I_ < I_gray_
	
	############################################
	# Finally, we try a dumb hard threshold :o #
	############################################
	I_ = 50 / 255
	I_thresholded = 1.0 * (I_ < I_gray)
	
	# Now we crop it.
	I_cropped = I_thresholded[(1-I_thresholded).sum(axis=1)>0,:][:,(1-I_thresholded).sum(axis=0)>0]
	
	display_image(I, title = 'Input image')
	display_image(I_gray, title = 'Grayscale image', cmap = 'gray')
	display_image(I_thresholded, title = 'Thresholded image', cmap = 'gray')
	display_image(I_cropped, title = 'Cropped image', cmap = 'gray')
	
	return I_cropped

def invertAndCeil(image):
	from skimage import util
	import numpy as np
	
	inverted_img = util.invert(image)
	inverted_img_ceil = np.ceil(inverted_img)
	
	return inverted_img_ceil

def digits(wild_image):
	import numpy as np
	
	from matplotlib import pyplot as plt
	plt.ion()
	
	cropped = crop(wild_image)
	resized = resize(cropped)
	inverted = invertAndCeil(resized)
	chopped = chop(inverted)
	
	display_image(resized, 'Resized image', 'gray')
	display_image(inverted, 'Inverted image', 'gray')
	
	n_digits = 10
	for i_digit, digit in enumerate(chopped):
		subplot = (1, n_digits, i_digit + 1)
		display_image(
			digit,
			'digits',
			'gray',
			subplot
		)
	
	show_images()
	
	n_images = chopped.shape[0]
	
	flattened = chopped.reshape(n_images, -1)
	
	return flattened
