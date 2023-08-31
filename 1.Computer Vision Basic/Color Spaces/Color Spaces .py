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


# In[4]:


image = cv2.imread(r'C:\Users\DELL\Desktop\python project\open cv2\New folder (21)\lisa.jpg')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Lisa')
plt.show()


# ### Color Spaces :

# Color spaces are mathematical models used to represent colors in a way that makes it easier to manipulate and analyze them. Different color spaces have different properties and are designed to suit specific purposes.

# #### RGB ( Red, Green, Blue ) :

# RGB is the most common color space used for electronic displays and digital imaging. It represents colors as combinations of red, green, and blue primary colors. Each pixel in an RGB image is made up of three values that indicate the intensity of each primary color. This color space is additive, meaning that mixing all three colors at full intensity produces white.

# In[21]:


blue_channel = image[:, :, 0]

blue_image = np.zeros_like(image)
blue_image[:, :, 0] = blue_channel

plt.imshow(cv2.cvtColor(blue_image, cv2.COLOR_BGR2RGB))
plt.title('Blue Lisa')
plt.show()
cv2.imwrite('blue_lisa.png', blue_image)


# In[20]:


red_channel = image[:, :, 2] 

red_image = np.zeros_like(image)
red_image[:, :, 2] = red_channel


plt.imshow(cv2.cvtColor(red_image, cv2.COLOR_BGR2RGB))
plt.title('Red Lisa')
plt.show()
cv2.imwrite('red_lisa.png', red_image)


# In[19]:


green_channel = image[:, :, 1]

green_image = np.zeros_like(image)
green_image[:, :, 1] = green_channel

plt.imshow(cv2.cvtColor(green_image, cv2.COLOR_BGR2RGB))
plt.title('Green Lisa')
plt.show()
cv2.imwrite('green_lisa.png', green_image)


# In[ ]:




