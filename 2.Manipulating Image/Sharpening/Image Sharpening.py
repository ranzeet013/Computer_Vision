#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


image = cv2.imread(r'C:\Users\DELL\Desktop\python project\open cv2\New folder (8)\pyramid.jpg')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.show()


# # Image Sharpening :

# Image sharpening is a technique used in image processing to enhance the clarity and details of an image, particularly its edges and fine structures. This is typically achieved through the application of a sharpening kernel using convolution. One common method for performing image sharpening is by using the cv2.filter2d function in the OpenCV library.

# In[4]:


kernel_sharpening = np.array([[-1, -1, -1], 
                              [-1, 9, -1], 
                              [-1, -1, -1]])
sharpened = cv2.filter2D(image, -1, kernel_sharpening)


# In[5]:


plt.imshow(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
plt.title('Sharpened Image')
plt.show()
cv2.imwrite("sharpened_image.png", sharpened)


# In[ ]:




