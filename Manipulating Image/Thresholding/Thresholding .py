#!/usr/bin/env python
# coding: utf-8

# # Thresholding 

# Thresholding is a fundamental image processing technique used to convert a grayscale image into a binary image. In thresholding, each pixel in the grayscale image is compared to a fixed threshold value. Depending on whether the pixel's intensity is above or below the threshold, the corresponding pixel in the binary image is set to either 0 (black) or 1 (white).

# If I(x, y) >= T, then B(x, y) = 1
# 
# If I(x, y) < T, then B(x, y) = 0

# In[2]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[5]:


image = cv2.imread(r'C:\Users\DELL\Desktop\python project\open cv2\prac 9\automobile.jpg')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Orginal Image')
plt.show()


# # Binary Thresholding :

# Binary thresholding is a simple image segmentation technique used to convert a grayscale image into a binary image, where pixels are either assigned the value 0 (black) or 1 (white) based on their intensity values. The threshold value determines the point at which the pixel intensity is classified as either 0 or 1.
# 
# If I(x, y) >= T, then B(x, y) = 1
# 
# If I(x, y) < T, then B(x, y) = 0

# In[35]:


ret, threshold1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
plt.imshow(cv2.cvtColor(threshold1, cv2.COLOR_BGR2RGB))
plt.title('Threshold Binary')
cv2.imwrite('binary_thresholding.png', threshold1)
plt.show()


# # Adaptive Thresholding :

# Adaptive thresholding is a variation of thresholding where the threshold value is not fixed for the entire image but varies locally based on the intensity values in the neighborhood of each pixel. This technique is particularly useful when the lighting conditions vary across the image or when there is significant variation in contrast.
# 
# If I(x, y) >= T(x, y), then B(x, y) = 1
# 
# If I(x, y) < T(x, y), then B(x, y) = 0

# In[36]:


image_blur = cv2.GaussianBlur(image, (3, 3), 0)
gray_blur = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)
threshold = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 5)
plt.imshow(threshold, cmap='gray')
plt.title('Adaptive Mean Thresholding')
cv2.imwrite('adaptive_thresholding.png', threshold)
plt.show()


# # Otsu's Thresholding :

# Otsu's thresholding, also known as maximum variance thresholding, is an automatic image thresholding technique that finds an optimal threshold value to separate the foreground and background pixels in an image. It assumes that the image contains two classes of pixels: the background and the foreground (objects of interest).
# 
# T = argmax(T) [ σ^2_B(T) / σ^2_W(T) ]

# In[39]:


gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, threshold2 = cv2.threshold(gray_image, 
                              0, 255, 
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.imshow(threshold2, cmap='gray')
plt.title("Otsu's Thresholding")
cv2.imwrite('otsu_thresholding.png', threshold2)
plt.show()


# # Gaussian Thresholding :

# In[43]:


blur = cv2.GaussianBlur(image, (5, 5), 0)
gray_blur = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
_, threshold3 = cv2.threshold(gray_blur, 
                              0, 255, 
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.imshow(threshold3, cmap='gray')
plt.title("Gaussian Thresholding")
cv2.imwrite('gaussian_thresholding.png', threshold3)
plt.show()

