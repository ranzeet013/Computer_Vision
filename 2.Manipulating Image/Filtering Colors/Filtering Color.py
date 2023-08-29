#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from matplotlib import pyplot as plt


# In[2]:


image = cv2.imread(r'C:\Users\DELL\Desktop\python project\open cv2\New folder\girl.jpg')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.show()


# ### Filtering Color :

# Filtering colors involves selectively allowing certain colors to pass through a filter while blocking others. This can be done using materials like color filters, specialized optical devices, or digital algorithms. It's used in photography, science, and digital image processing to create effects, correct colors, analyze specific wavelengths, and more.

# In[3]:


lower = np.array([90,0,0])
upper = np.array([135,255,255])

hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

mask = cv2.inRange(hsv_img, lower, upper)

res = cv2.bitwise_and(image, image, mask=mask)


# In[5]:


plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
plt.title('Masked Image')
plt.show()
cv2.imwrite('masked.png', mask)


# In[6]:


plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
plt.title('Filtered Image')
plt.show()
cv2.imwrite('filteres.png', res)


# In[ ]:




