#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


image = cv2.imread(r'C:\Users\DELL\Desktop\python project\open cv2\prac 14\note.jpg')
rows,cols,ch = image.shape
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Orginal Image')
plt.show()


# # Affine Transform :

# An affine transformation is a type of geometric transformation that preserves straight lines and parallelism between them. It is a fundamental concept in linear algebra and computer graphics. Affine transformations can be represented using a matrix multiplication and translation.

# In[8]:


point_A = np.float32([[120, 420], [525, 120], [720, 920]])
point_B = np.float32([[0, 0], [1280, 0], [0, 1280]])

metrics = cv2.getAffineTransform(point_A, point_B)

warped = cv2.warpAffine(image, metrics, (cols, rows))

plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
plt.title('warped_image')
plt.show()
cv2.imwrite('affine_warped.png', warped)


# In[ ]:




