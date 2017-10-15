def digits(wild_image):
	import crop
	import chop
	import numpy as np
	
	cropped = crop.crop(wild_image)
	chopped = chop.chop(cropped)
	
	import pdb ; pdb.set_trace()
	
	
	n_images = chopped.shape[0]
	
	flattened = np.reshape(n_images, -1)
	
	return flattened

def chop(cropped_image):
	import numpy as np
	
	imgparts = np.split(cropped_image, 10, 1)
	
	chopped_array = np.concatenate(imgparts)
	
	return imgparts

def resize(images):
	import skimage
	
	resized = [
		skimage.transfom.resize(image, [24, 240])
		for image in images
	]
	
	return resized

def crop(raw_image_path):
	import numpy as np
	import skimage as ski
	from skimage import io
	
	I = ski.io.imread(raw_image_path)
	
	# help(ski.transform.rescale) # DO THIS LATER. RESIZE INPUT IMAGE SO IT HAS UPPER BOUNDS ON SIZE
	
	#~ plt.imshow(I)
	#~ plt.title('Input image')
	
	I_gray = ski.color.rgb2gray(I)
	#~ plt.imshow(I_gray, cmap='gray')
	#~ plt.title('Grayscale image')
	# The thresholding methods I used below were borrowed/adapted from here http://scikit-image.org/docs/dev/auto_examples/xx_applications/plot_thresholding.html
	# from skimage.filters import threshold_local
	# local_block_size = 101
	# I_ = ski.filters.threshold_otsu(I_gray)
	# # I_ = ski.filters.threshold_local(I_gray, local_block_size, offset=50)
	# I_thresh = I_gray > I_
	# plt.imshow(I_thresh, cmap='gray')
	# # plt.imshow(I_, cmap='gray')
	# plt.title('Thresholded image')
	
	# from skimage.morphology import disk
	# from skimage.filters import threshold_otsu, rank
	# from skimage.util import img_as_ubyte
	
	# radius = 300
	# selem = disk(radius)
	
	# I_gray_ = ski.img_as_ubyte(I_gray)
	# I_ = rank.otsu(I_gray_, selem) # local otsu
	# # I_ = threshold_otsu(I_gray_)
	# # global_otsu = I_ < I_gray_
	# plt.imshow(I_, cmap='gray')
	# plt.title('Pre-thresholding image')
	
	############################################
	# Finally, we try a dumb hard threshold :o #
	############################################
	I_ = 50
	I_thresholded = I_ < I_gray
	#~ plt.imshow(I_thresholded, cmap='gray')
	#~ plt.title('Thresholded image')
	
	# Now we crop it.
	I_cropped = I_thresholded[(1-I_thresholded).sum(axis=1)>0,:][:,(1-I_thresholded).sum(axis=0)>0]
	#~ plt.imshow(I_cropped, cmap='gray')
	
	return I_cropped
