#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[6]:


image = cv2.imread(r'C:\Users\DELL\Desktop\python project\open cv2\prac 7\computer.jpg')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Computer Image')
plt.show()


# # Canny Edge :

# Canny edge detection is a popular image processing technique used to detect edges in digital images. It has become a standard algorithm for edge detection due to its effectiveness and robustness.The main idea behind the Canny edge detection algorithm is to find areas in the image with rapid intensity changes, which are likely to correspond to edges.

# In[7]:


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edge = cv2.Canny(gray, 30, 200)
plt.imshow(cv2.cvtColor(edge, cv2.COLOR_BGR2RGB))
plt.title('Canny Edge')
cv2.imwrite('canny_edge.png', edge)


# # Contours : 

# Contours are the continuous curves or outlines that represent the boundaries of objects or shapes in an image. They are an essential concept in image processing and computer vision and are often used for object detection, recognition, and shape analysis.There are various methods to extract contours from an image, and one common approach is to use edge detection algorithms, like the Canny edge detection mentioned earlier, to find the edges of objects. Once edges are detected, the contours are determined by connecting the adjacent edge points.

# In[8]:


contour, hierary = cv2.findContours(edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
print("Contour : " +str(len(contour)))


# In[9]:


cv2.drawContours(image, contour, -1, (0, 0, 255), 3)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Contour Image')
cv2.imwrite("contour_image.png", image)


# In[ ]:




