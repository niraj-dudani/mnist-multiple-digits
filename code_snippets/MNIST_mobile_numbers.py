
# coding: utf-8

# In[36]:


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


# In[37]:


img1=mpimg.imread('phonenum1.jpeg')
img2=mpimg.imread('phonenum2.jpeg')
img3=mpimg.imread('phonenum3.jpeg')
img4=mpimg.imread('phonenum4.jpeg')


# In[49]:


grayimg1 = np.dot(img1, [0.299, 0.587, 0.114])
grayimg2 = np.dot(img2, [0.299, 0.587, 0.114])
grayimg3 = np.dot(img3, [0.299, 0.587, 0.114])
grayimg4 = np.dot(img4, [0.299, 0.587, 0.114])


# In[50]:


imgparts1 = np.split(grayimg1, 10, 1)
imgparts2 = np.split(grayimg2, 10, 1)
imgparts3 = np.split(grayimg3, 10, 1)
imgparts4 = np.split(grayimg4, 10, 1)

