#!/usr/bin/env python
# coding: utf-8

# # Importing Image

# Importing Image using cv2:

# In[1]:


import cv2
import numpy as np


# In[2]:


image = cv2.imread(r'C:\Users\DELL\Desktop\python project\open cv2\transformation\iroha.jpg', cv2.IMREAD_COLOR)


# In[3]:


def display(winame, image):
    cv2.imshow(winame, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# In[4]:


display('iroha', image)


# # Image Shifting

# The image shifting project aims to develop a system that can shift images by a specified number of pixels in different directions using the OpenCV library (cv2). The project leverages computer vision techniques to manipulate the spatial position of pixels in an image, allowing for translation effects

# # Shifting Image 100px Right

# To shift an image 100 pixels to the right. It loads the image, defines the shifting amount, creates a transformation matrix, applies the transformation, and displays the original and shifted images.

# In[5]:


tx = 100
ty = 100
imag = np.float32([[1, 0, tx], 
            [0, 1, ty]])


# In[7]:


imag


# In[8]:


shift_right = cv2.warpAffine(image, imag, (image.shape[1], image.shape[0]))


# In[9]:


display('Right Shifted Iroha Image', shift_right)


# # Shifting Image 100px Left

# To shift an image 100 pixels to the left. It loads the image, defines the shifting amount, creates a transformation matrix, applies the transformation, and displays the original and shifted images.

# In[16]:


def shift_left(image, tx, ty):
    img = np.float32([[1, 0, tx], 
                      [0, 1, ty]])
    shifted_image = cv2.warpAffine(image, img, (image.shape[1], image.shape[0]))
    display('Left Shifted Iroha Image', shifted_image)


# In[19]:


shift_left(image, -100, 100)

