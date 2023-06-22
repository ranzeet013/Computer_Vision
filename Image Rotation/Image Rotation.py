#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# OpenCV (cv2) is widely used in fields such as computer vision, image and video processing, machine learning, and robotics. It provides a comprehensive set of functions and algorithms for tasks such as image manipulation, feature detection, object recognition, image filtering, camera calibration, and more.

# In[1]:


import cv2
import numpy as np


# # Displaying Image

# In[28]:


img = cv2.imread(r'C:\Users\DELL\Desktop\python project\open cv2\image rotation\iroha.jpg')


# In[4]:


cv2.imshow('Orginal Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # Rotating Image

# The Image Rotator project aims to develop a software application that allows users to rotate images by a specified angle. The project will provide a user-friendly interface where users can load an image file, select the desired rotation angle, and apply the rotation to the image. The rotated image can then be saved to a new file.

# In[5]:


image_center = (image.shape[1]//2, image.shape[0]//2)


# In[6]:


rotation_matrix = cv2.getRotationMatrix2D(image_center, 90, 1)


# In[7]:


rotation_matrix


# In[9]:


rotate_90 = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))


# In[11]:


display('Rotated 90', rotate_90)


# In[14]:


cv2.imshow('Rotation 90', rotate_90)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # Rotation Function

# Here's a function that rotates an image using the OpenCV library (cv2) in Python:

# In[36]:


def rotate(image, angle, scale):
    center = (image.shape[1]//2, image.shape[0]//2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotate_img = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    
    cv2.imshow('Rotated Image', rotate_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# In[42]:


rotate(img, 30, 1)

