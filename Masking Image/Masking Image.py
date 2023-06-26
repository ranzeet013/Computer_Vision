#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# cv2 is the OpenCV library for image processing and manipulation.
# 
# numpy (abbreviated as np) is a library for numerical operations, often used for array manipulation.
# 
# matplotlib.pyplot (abbreviated as plt) is a library for creating visualizations, including displaying images.

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# # Displaying Image

# In[2]:


image = cv2.imread(r'C:\Users\DELL\Desktop\python project\open cv2\masking image\iroha.jpg')


# In[3]:


cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # Masking Image

# Image masking is a technique used to selectively hide or reveal parts of an image based on a binary mask. The mask consists of black and white pixels, where black pixels indicate the areas to be hidden and white pixels indicate the areas to be kept. By applying the mask to the image, you can effectively hide specific regions while preserving the rest of the image.

# In[4]:


mask = np.zeros(image.shape[:2], dtype = 'uint8')
cv2.imshow('masking ', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[5]:


cx, cy = mask.shape[1]//2, mask.shape[0]//2

cv2.rectangle(mask, (cx-200, cy-200), (cx+200, cy+200), 255, -1)

cv2.imshow('mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[6]:


not_mask = cv2.bitwise_not(mask)

cv2.imshow('not_masked', not_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[8]:


mask_image = cv2.bitwise_and(image, image, mask = mask)

cv2.imshow('Orginal Image', image)
cv2.imshow('Mask Image', mask_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[9]:


image_not_masked = cv2.bitwise_and(image, image, mask = not_mask)

cv2.imshow('Orginal Image', image)
cv2.imshow('Not_Mask Image', not_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

