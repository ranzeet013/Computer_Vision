#!/usr/bin/env python
# coding: utf-8

# # Brightening Image

# In this project, the goal is to develop a program that can brighten images by adjusting the pixel values. The program will take an input image and apply a brightness adjustment to make the image appear brighter.

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


add_values = np.ones_like(image, dtype = 'uint8')*50


# In[5]:


cv2.imshow('add_values', add_values)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[6]:


add_image = cv2.add(image, add_values)


# In[8]:


cv2.imshow('original_image', image)
cv2.imshow('add_image', add_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

