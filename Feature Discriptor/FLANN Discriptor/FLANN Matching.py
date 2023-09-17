#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


lays_image = cv2.imread(r'C:\Users\DELL\Desktop\python project\open cv2\New folder (18)\blue.jpg')
plt.imshow(cv2.cvtColor(lays_image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.show()


# In[3]:


snack_image = cv2.imread(r'C:\Users\DELL\Desktop\python project\open cv2\New folder (18)\lays.jpg')
plt.imshow(cv2.cvtColor(snack_image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.show()


# ### FLANN Matching :

# FLANN, or Fast Library for Approximate Nearest Neighbors, is a crucial tool for computer vision. It efficiently matches features between images, vital for tasks like image stitching and object recognition. FLANN rapidly finds the most similar descriptors, streamlining the process and improving task accuracy. Its applications extend to image retrieval and various machine learning tasks, making it a fundamental component in computer vision workflows. 

#  I'm using OpenCV to perform feature matching between two images. I start by initializing the SIFT detector and then use it to find keypoints and descriptors in both images. Next, I set up FLANN parameters for efficient matching. The FLANN-based matcher is created, and I find potential matches between keypoints in the images. To ensure high-quality matches, I apply a ratio test and create a mask to filter the best matches. Finally, I draw these good matches on a new image, providing a visual representation of the corresponding keypoints. This code is valuable for tasks like image stitching and object recognition, where matching features are essential for accurate results.

# In[4]:


sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(lays_image,None)
kp2, des2 = sift.detectAndCompute(snack_image,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)  

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# ratio test
for i,(match1,match2) in enumerate(matches):
    if match1.distance < 0.7*match2.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

flann_matches = cv2.drawMatchesKnn(lays_image,kp1,snack_image,kp2,matches,None,**draw_params)


# In[5]:


plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(flann_matches, cv2.COLOR_BGR2RGB))
plt.title("Matching Image")
plt.show()
cv2.imwrite('flann_discriptor.png', flann_matches)


# In[ ]:




