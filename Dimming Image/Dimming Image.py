#!/usr/bin/env python
# coding: utf-8

# # Dimming Image

# In this project, the aim is to develop a program that can reduce the brightness of images by adjusting the pixel values. The program will take an input image and apply a darkening effect to make the image appear darker. 

# In[1]:


import cv2
import numpy as np


# In[2]:


image = cv2.imread(r'C:\Users\DELL\Desktop\python project\open cv2\image transform\iroha.jpg', cv2.IMREAD_COLOR)


# In[3]:


cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[4]:


sub_values = np.ones_like(image, dtype = 'uint8')*75


# In[5]:


cv2.imshow('sub_values', sub_values)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[6]:


sub_image = cv2.subtract(image, sub_values)


# In[7]:


cv2.imshow('original_image', image)
cv2.imshow('sub_image', sub_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

