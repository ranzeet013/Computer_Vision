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


# In[3]:


image = cv2.imread(r'C:\Users\DELL\Desktop\python project\open cv2\New folder (12)\sunflowers.jpg')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.show()


# ### Blob Detection :

# The cv2.drawKeypoints function in OpenCV is designed to visualize keypoints detected in an image. It accepts the original image, a list of keypoints, an output array, a color, and flags as inputs. These flags, like DRAW_MATCHES_FLAGS_DEFAULT and DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, control the drawing behavior. For instance, the "rich keypoints" flag enhances the visualization by displaying circles with size and orientation information. This function aids in understanding keypoint-based feature detection outcomes by overlaying keypoints onto imag

# In[4]:


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
detector = cv2.SimpleBlobDetector_create()
keypoints = detector.detect(image)
blank = np.zeros((1,1)) 
blobs = cv2.drawKeypoints(gray, keypoints, blank, (255,0,255),
                                      cv2.DRAW_MATCHES_FLAGS_DEFAULT)
plt.imshow(cv2.cvtColor(blobs, cv2.COLOR_BGR2RGB))
plt.title('Output_Blobs')
plt.show()
cv2.imwrite('output_blobs.jpg', blobs)


# In[ ]:





# In[ ]:




