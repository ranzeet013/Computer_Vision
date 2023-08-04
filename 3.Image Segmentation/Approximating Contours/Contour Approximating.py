#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


image = cv2.imread(r'C:\Users\DELL\Desktop\python project\open cv2\prac 17\house.jpg')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Orginal Image')
plt.show()


# # Gray Image :

# In[4]:


gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB))
plt.title('Gray Image')
plt.show()
cv2.imwrite('Gray_Image.png', gray_image)


# # Thresholding :

# In[9]:


ret, thresh = cv2.threshold(gray_image, 130, 255, cv2.THRESH_BINARY_INV)
plt.imshow(cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB))
plt.title('Threshold')
plt.show()
cv2.imwrite('Thresholding.png', thresh)


# # Approximating Contours :

# In[18]:


contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
orig_image = image.copy()
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(orig_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    plt.imshow(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))
    plt.title('Bounding Rectangle')
    plt.show()
    


# In[ ]:




