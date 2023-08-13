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


image = cv2.imread(r'C:\Users\DELL\Desktop\python project\open cv2\heatmap\iroha.jpg')       #image path


# In[5]:


cv2.imshow('orginal image', image)                                                           #displaying image
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[9]:


image.shape


# # Convering To GrayImage

# In[6]:


gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                                         #converting image to grayscale


# In[8]:


cv2.imshow('gray_image', gray_image)                                                         #displaying gray image
cv2.imshow('Orginal_image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[10]:


gray_image.shape


# # Heatmap Visualization

# In[12]:


heatmap = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)                               #applying color map in image


# In[13]:


heatmap_image = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)                            #defining the weighted parameters


# In[15]:


cv2.imshow('Heatmap Visualization', heatmap_image)                                      #visualizing heat map image
cv2.imshow('orginal_image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

