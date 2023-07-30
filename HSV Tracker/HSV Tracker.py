#!/usr/bin/env python
# coding: utf-8

# # HSV Tracker

# An HSV tracker is a computer vision algorithm that tracks objects in images or videos based on their color in the HSV color space. It works by extracting the HSV values of the target object in the first frame and then finding similar color pixels in subsequent frames to track the object's movement. It is effective for objects with consistent and distinct colors but may face challenges with color changes or similar-looking objects in complex scenes.

# In[2]:


import cv2
import numpy as np
def nothing(x):
    pass

cv2.namedWindow("Trackbars")

cv2.createTrackbar("L-H","Trackbars",0,179,nothing)
cv2.createTrackbar("L-S","Trackbars",0,255,nothing)
cv2.createTrackbar("L-V","Trackbars",0,255,nothing)
cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, frame = cap.read()
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h= cv2.getTrackbarPos("L-H,","Trackbars")
    l_s= cv2.getTrackbarPos("L-S,","Trackbars")
    l_v= cv2.getTrackbarPos("L-V,","Trackbars")


    lower_blue = np.array([l_h,l_s,l_v])
    upper_blue = np.array([u_h,u_s,u_v])
    
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(frame,frame, mask= mask)

    cv2.imshow('frame',frame)
    cv2.imshow('res',res)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()


# In[ ]:




