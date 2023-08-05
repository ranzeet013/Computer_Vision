#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


image = cv2.imread(r'C:\Users\DELL\Desktop\python project\open cv2\prac 12\high_five.jpg')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Orginal Image')
plt.show()


# # Convex Hull :

# Convex hull is the smallest convex shape that encloses a given set of points. It can be visualized as the shape formed by stretching a rubber band around the outermost points. There are efficient algorithms to compute the convex hull, and it has applications in computer graphics, computer vision, and robotics.

# In[4]:


gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray_image, 176, 255, 0)

contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
number = len(contours) - 1
contours = sorted(contours, key=cv2.contourArea, reverse=False)[:number]


# In[5]:


for c in contours:
    hull = cv2.convexHull(c)
    cv2.drawContours(image, [hull], 0, (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Convex Hull')
    plt.show()
    cv2.imwrite('convexHull.jpg', image)


# In[ ]:




