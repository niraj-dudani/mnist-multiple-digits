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

def auto_crop(I):
	"""Find bounding box of image, crop to this box, and return a view
	of this (use numpy.copy on the returned object, if needed).
	Assumes background pixels have value '1'."""
	
	import numpy as np
	
	row_sum = (1 - I).sum(axis=1)
	col_sum = (1 - I).sum(axis=0)
	top = np.nonzero(row_sum)[0][0]
	bottom = np.nonzero(row_sum)[0][-1]
	left = np.nonzero(col_sum)[0][0]
	right = np.nonzero(col_sum)[0][-1]
	
	# print(left, right, top, bottom)
	return I[top:bottom,left:right]

def chop(cropped_image):
	import numpy as np
	import scipy as sp
	from scipy import signal
	import skimage
	import skimage.transform
	
	intensities_horz = cropped_image.sum(axis=0)
	
	# we seem to need to pad it, for wavelet transform based peak detection to 
	# work properly
	
	intensities_horz = np.pad(
		intensities_horz,
		20,
		mode = 'constant',
		constant_values = 1 * cropped_image.shape[0]
	)
	
	#~ plt.plot(intensities_horz)
	peak_locs = sp.signal.find_peaks_cwt(intensities_horz,np.arange(20,70))
	peak_locs = np.maximum(np.array(peak_locs) - 20, 0)
	
	digits=[]
	for i in range(0, len(peak_locs) - 1):    
		digit = cropped_image[:, peak_locs[i]:peak_locs[i+1]];
		digit = auto_crop(digit).copy()
		
		# resize it
		s = np.array(digit.shape);
		s = ((24.0*s/s.max()).round().astype('int'))
		digit = skimage.transform.resize(
			digit,
			s,
			order = 3,
			mode = 'constant'
		)
		
		# pad it
		pad_l = np.floor((28 - s[0])/2).astype('int')
		pad_r = 28 - s[0] - pad_l
		pad_t = np.floor((28 - s[1])/2).astype('int')
		pad_b = 28 - s[1] - pad_t
		# print ([pad_l,pad_r],[pad_t,pad_b])
		digit = np.pad(
			digit,
			[[pad_l,pad_r],[pad_t,pad_b]],
			mode='constant',
			constant_values = 1
		)
		
		# print(digit.shape)
		
		digits.append(digit)
	
	n_digits = len(digits)
	n_cols = digits[0].shape[1]
	
	chopped_array = np.concatenate(digits)
	chopped_array = chopped_array.reshape(n_digits, -1, n_cols)
	
	return chopped_array

def chop_uniform(cropped_image):
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

def read_image(raw_image_path):
	import skimage.io
	
	I = skimage.io.imread(raw_image_path)
	return I

def desaturate(image):
	import skimage
	
	I_gray = skimage.color.rgb2gray(image)
	
	return I_gray

def threshold(grayscale_image):
	import skimage
	
	I_ = 50 / 255
	I_thresholded = 1.0 * (I_ < grayscale_image)
	
	return I_thresholded

def crop_monolith(raw_image_path):
	import numpy as np
	import skimage
	import skimage.io
	
	I = skimage.io.imread(raw_image_path)
	
	# help(skimage.transform.rescale) # DO THIS LATER. RESIZE INPUT IMAGE SO IT HAS UPPER BOUNDS ON SIZE
	
	I_gray = skimage.color.rgb2gray(I)
	
	# The thresholding methods I used below were borrowed/adapted from here http://scikit-image.org/docs/dev/auto_examples/xx_applications/plot_thresholding.html
	# from skimage.filters import threshold_local
	# local_block_size = 101
	# I_ = skimage.filters.threshold_otsu(I_gray)
	# # I_ = skimage.filters.threshold_local(I_gray, local_block_size, offset=50)
	# I_thresh = I_gray > I_
	
	# from skimage.morphology import disk
	# from skimage.filters import threshold_otsu, rank
	# from skimage.util import img_as_ubyte
	
	# radius = 300
	# selem = disk(radius)
	
	# I_gray_ = skimage.img_as_ubyte(I_gray)
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

def digits(wild_image_path):
	import numpy as np
	from matplotlib import pyplot as plt
	plt.ion()
	
	input_image = read_image(wild_image_path)
	grayscale_image = desaturate(input_image)
	thresholded_image = threshold(grayscale_image)
	cropped_image = auto_crop(thresholded_image)
	#~ resized_image = resize(cropped_image)
	inverted_image = invertAndCeil(cropped_image)
	chopped_images = chop(inverted_image)
	
	display_image(input_image, title = 'Input image')
	display_image(grayscale_image, title = 'Grayscale image', cmap = 'gray')
	display_image(thresholded_image, title = 'Thresholded image', cmap = 'gray')
	display_image(cropped_image, title = 'Cropped image', cmap = 'gray')
	#~ display_image(resized_image, 'Resized image', 'gray')
	display_image(inverted_image, 'Inverted image', 'gray')
	
	n_digits = 10
	for i_digit, digit in enumerate(chopped_images):
		subplot = (1, n_digits, i_digit + 1)
		display_image(
			digit,
			'digits',
			'gray',
			subplot
		)
	
	show_images()
	
	n_images = chopped_images.shape[0]
	
	flattened = chopped_images.reshape(n_images, -1)
	
	return flattened
