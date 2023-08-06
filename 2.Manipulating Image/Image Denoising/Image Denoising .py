#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


image = cv2.imread(r'C:\Users\DELL\Desktop\python project\open cv2\New folder\tiger.jpg')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Orginal Image')
plt.show()


# # Denoising :

# Fast means denoising is a technique used to remove noise from images. It works by dividing the image into small patches, finding similar patches based on pixel intensity similarity, and then averaging those similar patches to reduce noise. OpenCV provides the cv2.fastNlMeansDenoisingColored function for denoising colored images and cv2.fastNlMeansDenoising for grayscale images. It is computationally efficient and suitable for real-time applications and larger images.

# In[4]:


denoised = cv2.fastNlMeansDenoisingColored(image, None, 11, 6, 7, 21)
plt.imshow(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
plt.title('Fast Means Denosisng')
plt.show()
cv2.imwrite('Denoised Image.png', denoised)


# In[ ]:




