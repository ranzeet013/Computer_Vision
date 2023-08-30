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


image = cv2.imread(r'C:\Users\DELL\Desktop\python project\open cv2\New folder (7)\board.jpg')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.show()


# ### Corner Detection :

# Corner detection using the Harris Corner Detection algorithm spots key points in images where intensity changes occur in multiple directions. By evaluating local gradients, it distinguishes corners from edges. Strong corner responses are highlighted after dilation, helping in identifying significant features for various computer vision tasks.

# Image is loaded in grayscale and converted to floating-point format. The Harris Corner Detection algorithm is then applied to find significant corners in the image. The detected corner responses are enhanced by dilation. A threshold is applied to highlight prominent corners, which are marked with a distinctive color. The resulting image is displayed using matplotlib, showing the identified corners. Additionally, the corner-detected image is saved as 'corner_detected.png'. 

# In[4]:


image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
image = np.float32(image)

blockSize = 2
ksize = 3
k = 0.04
dst = cv2.cornerHarris(image, blockSize, ksize, k)
dst = cv2.dilate(dst, None)

threshold = 0.01 * dst.max()
corner_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
corner_image[dst > threshold] = [0, 255, 255] 

plt.imshow(cv2.cvtColor(corner_image, cv2.COLOR_BGR2RGB))
plt.title('Corner Detected')
plt.show()
cv2.imwrite('corner_detected.png', corner_image)


# In[ ]:




