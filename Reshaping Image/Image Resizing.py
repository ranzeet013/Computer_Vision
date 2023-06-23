#!/usr/bin/env python
# coding: utf-8

# # Resizing Image

# To resize an image using the OpenCV library in Python, we can make use of the cv2.resize() function. This function allows us to specify the desired size for the output image.

# In[1]:


import cv2
import numpy as np


# In[5]:


image = cv2.imread(r'C:\Users\DELL\Desktop\python project\open cv2\reshizing image\iroha.jpg')


# In[8]:


cv2.imshow('Oregeiru', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[9]:


image.shape


# In[10]:


image_resize = cv2.resize(image, (350, 350), interpolation = cv2.INTER_AREA)


# In[12]:


cv2.imshow('Resized Image', image_resize)
cv2.imshow('Oregairu', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

