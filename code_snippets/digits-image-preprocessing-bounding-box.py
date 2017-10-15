
# coding: utf-8

# We preprocess images which have handwritten digits. We find the bounding box first.

# In[181]:


get_ipython().magic(u'matplotlib inline')

import numpy as np
import skimage as ski
from skimage import io


# In[182]:


I = ski.io.imread('./data/WhatsApp Image 2017-10-14 at 3.18.01 PM.jpeg') # is clean
# I = ski.io.imread('./data/WhatsApp Image 2017-10-14 at 3.18.00 PM.jpeg') # has an unwanted huge shadow

# help(ski.transform.rescale) # DO THIS LATER. RESIZE INPUT IMAGE SO IT HAS UPPER BOUNDS ON SIZE


# In[183]:


plt.imshow(I)
plt.title('Input image')


# In[184]:


I_gray = ski.color.rgb2gray(I)
plt.imshow(I_gray, cmap='gray')
plt.title('Grayscale image')


# The thresholding methods I used below were borrowed/adapted from here http://scikit-image.org/docs/dev/auto_examples/xx_applications/plot_thresholding.html

# In[185]:


# from skimage.filters import threshold_local


# In[186]:


# local_block_size = 101
# I_ = ski.filters.threshold_otsu(I_gray)
# # I_ = ski.filters.threshold_local(I_gray, local_block_size, offset=50)
# I_thresh = I_gray > I_
# plt.imshow(I_thresh, cmap='gray')
# # plt.imshow(I_, cmap='gray')
# plt.title('Thresholded image')


# In[187]:


# from skimage.morphology import disk
# from skimage.filters import threshold_otsu, rank
# from skimage.util import img_as_ubyte


# In[188]:


# radius = 300
# selem = disk(radius)

# I_gray_ = ski.img_as_ubyte(I_gray)
# I_ = rank.otsu(I_gray_, selem) # local otsu
# # I_ = threshold_otsu(I_gray_)
# # global_otsu = I_ < I_gray_
# plt.imshow(I_, cmap='gray')
# plt.title('Pre-thresholding image')


# In[189]:


############################################
# Finally, we try a dumb hard threshold :o #
############################################
I_ = 50
I_thresholded = I_ < I_gray_
plt.imshow(I_thresholded, cmap='gray')
plt.title('Thresholded image')


# Now we crop it.

# In[191]:


I_cropped = I_thresholded[(1-I_thresholded).sum(axis=1)>0,:][:,(1-I_thresholded).sum(axis=0)>0]
plt.imshow(I_cropped, cmap='gray')

