#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[5]:


lays = cv2.imread(r'C:\Users\DELL\Desktop\python project\open cv2\New folder (18)\lays.jpg')
plt.imshow(cv2.cvtColor(lays, cv2.COLOR_BGR2RGB))
plt.title('Lays Image')
plt.show()


# In[6]:


lays_blue = cv2.imread(r'C:\Users\DELL\Desktop\python project\open cv2\New folder (18)\blue.jpg')
plt.imshow(cv2.cvtColor(lays_blue, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.show()


# ### Matching With SIFT Discriptor :

# I'm using the OpenCV library to work with images and perform a task involving keypoint matching. I start by initializing the SIFT (Scale-Invariant Feature Transform) detector using cv2.xfeatures2d.SIFT_create(). This detector helps identify distinctive points in images, known as keypoints, and computes descriptors that capture local image information around these keypoints.
# 
#  I apply the SIFT detector to two input images, which I've named 'lays' and 'lays_blue'. For each image, I detect keypoints and compute descriptors, resulting in kp1, des1 for 'lays' and kp2, des2 for 'lays_blue'.
#  
#  I then set up a Brute-Force Matcher using cv2.BFMatcher(). This matcher is going to be helpful for comparing keypoints between the two images based on their descriptors.With the matcher ready, I perform K-Nearest Neighbor (KNN) matching using bf.knnMatch(des1, des2, k=2). This step compares the descriptors of keypoints from 'lays' to those from 'lays_blue', finding the two best matches for each descriptor in 'lays'.

# In[8]:


sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(lays,None)
kp2, des2 = sift.detectAndCompute(lays_blue,None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

good = []
for match1,match2 in matches:
    if match1.distance < 0.75*match2.distance:
        good.append([match1])

sift_matches = cv2.drawMatchesKnn(lays, kp1 ,lays_blue ,kp2 ,good, None, flags=2)


# In[11]:


plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(sift_matches, cv2.COLOR_BGR2RGB))
plt.title("Matching Image")
plt.show()
cv2.imwrite('SIFT_discriptor.png', sift_matches)


# In[ ]:




