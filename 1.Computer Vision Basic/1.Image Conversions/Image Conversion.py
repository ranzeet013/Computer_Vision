#!/usr/bin/env python
# coding: utf-8

# # Displaying Image

# In[1]:


import cv2
import numpy as np


# In[2]:


image = cv2.imread(r'C:\Users\DELL\Desktop\python project\open cv2\New folder\iroha.jpg')


# In[3]:


cv2.imshow('Iroha And Yuighama', image)                   #orginal image


# In[4]:


cv2.waitKey(0)
cv2.destroyAllWindows()


# In[17]:


image.shape


# # Converting To GrayScale

# Grayscale refers to an image or display mode that represents shades of gray or black and white, rather than color. In a grayscale image, each pixel is assigned a gray value ranging from 0 (black) to 255 (white), with various shades of gray in between.

# In[5]:


gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                       #converting to gray image


# In[8]:


cv2.imshow('Iroha And Yuigahama', image)
cv2.imshow('Gray Image', gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[16]:


gray_image.shape


# # Gaussian Blur

# Gaussian blur is a popular image processing technique used to reduce image noise and smooth out details by applying a specific type of blur known as a Gaussian function.

# In[10]:


gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)                    #converting to gaussian blur 


# In[11]:


cv2.imshow('Iroha And Yui', image)
cv2.imshow('Gaussian Blur Image', gaussian_blur)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[15]:


gaussian_blur.shape


# # Division Image

# In[18]:


division_image = cv2.divide(image, gaussian_blur, scale = 256)          #converting to division image


# In[19]:


cv2.imshow('Iroha And Yui', image)
cv2.imshow('Divison Image', division_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # Adjusting Gamma Factor

# The gamma factor, often denoted as γ (gamma), is a parameter used to adjust the brightness or contrast of digital images. It controls the relationship between the pixel values of the original image and the displayed or processed output.

# Output = Input^γ

# In[25]:


gamma = 0.1                                   #adjusting value of gamma
if gamma == 0:                                #keeping gamma value to smallest value
    gamma = 0.01              

invgamma = 1/gamma                            #calculating inverse gamma
look_up_table = np.array([((i/255)**invgamma)*255 for i in range (0, 256)])         #normalizing the gamma value 

pencil_sketch = cv2.LUT(division_image.astype('uint8'), look_up_table.astype('uint8'))       #unidentifiedint8 in use


# In[26]:


cv2.imshow('Iroha And Yui', image)
cv2.imshow('Pencil_Sketch', pencil_sketch)
cv2.waitKey(0)
cv2.destroyAllWindows()

