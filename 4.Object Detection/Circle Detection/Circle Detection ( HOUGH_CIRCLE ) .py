#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[5]:


image = cv2.imread(r'C:\Users\DELL\Desktop\python project\open cv2\New folder (11)\circle.jpg')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.show()


# ### Circle Detection ( HOUGH_CIRCLE )  :

# The cv2.HoughCircles function in OpenCV is used to detect circles in images using the Hough Circle Transform method. It requires parameters like method (usually cv2.HOUGH_GRADIENT), dp for resolution control, MinDist to set minimum circle distance, param1 for edge detection, param2 for detection threshold, and minRadius and MaxRadius to limit circle sizes. By adjusting these parameters, you can fine-tune circle detection for different scenarios, aiding applications like object recognition and automation.

# In[9]:


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(gray, 5)

circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1.2, 25)
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    
    for i in circles:
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
    
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Detected Circles')
    plt.show()
else:
    print("No circles detected.")


# In[ ]:




