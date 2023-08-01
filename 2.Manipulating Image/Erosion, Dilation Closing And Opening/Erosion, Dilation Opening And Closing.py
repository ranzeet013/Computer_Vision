#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


image = cv2.imread(r'C:\Users\DELL\Desktop\python project\open cv2\prac 8\sutterlin.jpg')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Orginal Image')
plt.show()


# # Erosion :

# Erosion is performed by placing the center of the structuring element at each pixel of the input image and setting the pixel value at that position in the eroded image to 1 (white) only if all corresponding pixels in the structuring element and the input image are 1. Otherwise, the pixel value in the eroded image is set to 0 (black).

# E(A) = A ⊖ B
# 
# Where:
# 
# E(A) is the eroded image.
# 
# A is the input binary image.
# 
# B is the structuring element (also known as the kernel or mask).

# In[6]:


kernel = np.ones((5, 5), np.uint8)


# In[10]:


erosion = cv2.erode(image, kernel, iterations = 1)
plt.imshow(cv2.cvtColor(erosion, cv2.COLOR_BGR2RGB))
plt.title('Erosion Image')
cv2.imwrite('Erosion_image.png', erosion)


# # Dilation :

# Dilation sets a pixel in the output image to 1 (white) if at least one corresponding pixel in the structuring element and the input image is 1. Otherwise, it remains 0 (black). This operation is helpful for image preprocessing and feature extraction tasks in computer vision.
# 
# D(A) = A ⊕ B
# 
# Where:
# 
# D(A) is the dilated image.
# 
# A is the input binary image.
# 
# B is the structuring element (kernel or mask).

# In[12]:


dilation = cv2.dilate(image, kernel, iterations = 1)
plt.imshow(cv2.cvtColor(dilation, cv2.COLOR_BGR2RGB))
plt.title('Dilation Image')
cv2.imwrite('Dilation_image.png', dilation)


# # Opening :

# Opening is a morphological operation that removes small noise and unwanted elements from a binary image. It involves performing erosion followed by dilation using a structuring element. Opening helps to smooth the image and preserve the main object shapes while eliminating noise and thin connections between objects.
# 
# O(A) = (A ⊖ B) ⊕ B
# 
# Where:
# 
# O(A) is the output image after the opening operation
# 
# A is the input binary image.
# 
# B is the structuring element (kernel or mask).

# In[13]:


opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
plt.imshow(cv2.cvtColor(opening, cv2.COLOR_BGR2RGB))
plt.title('Opening Image')
cv2.imwrite('Opening_image.png', opening)


# # Closing :

# The closing operation helps in smoothing the image, filling small holes or gaps, and connecting broken parts of the objects. It is useful for image preprocessing, noise reduction, and feature extraction tasks in computer vision.
# 
# C(A) = (A ⊕ B) ⊖ B
# 
# Where:
# 
# C(A) is the output image after the closing operation.
# 
# A is the input binary image.
# 
# B is the structuring element (kernel or mask).

# In[16]:


closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
plt.imshow(cv2.cvtColor(closing, cv2.COLOR_BGR2RGB))
plt.title('Closing Image')
cv2.imwrite('closing_image.png', closing)

