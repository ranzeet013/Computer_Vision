#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


image = cv2.imread(r'C:\Users\DELL\Desktop\python project\open cv2\prac 13\Sudoku.jpg')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Orginal Image')
plt.show()


# # Perspective Transform

# Perspective transform is a geometric transformation used to convert the perspective view of an object or image into a new view. It is often applied to correct distortions caused by the perspective projection when capturing images or to warp images into a different perspective.

# The perspective transform requires four points from the source image and four corresponding points in the destination image to calculate the transformation matrix. These points form a quadrilateral in both the source and destination images. By applying the transformation matrix, the source image can be warped to align with the destination quadrilateral, effectively changing its perspective.

# In[6]:


point_A = np.float32([[28, 29], [570, 30], [25, 570], [570, 570]])
point_B = np.float32([[0, 0], [600, 0], [0, 600], [600, 600]])

metrix = cv2.getPerspectiveTransform(point_A, point_B)

warped = cv2.warpPerspective(image, metrix, (600, 600))
plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
plt.title('Perspective Transform ')
plt.show()
cv2.imwrite('Warped_image.png', warped)


# In[ ]:




