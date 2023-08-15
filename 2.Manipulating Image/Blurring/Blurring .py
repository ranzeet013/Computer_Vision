#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


image = cv2.imread(r'C:\Users\DELL\Desktop\python project\open cv2\New folder (7)\leopard.jpg')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.show()


# # Blurring :

# Blurring, also known as image smoothing or filtering, is a common image processing technique used to reduce noise, remove fine details, and create a smoother appearance in images. Blurring is achieved by averaging the pixel values in the neighborhood of each pixel, which results in a "blurred" or smoothed version of the image.

# In[7]:


kernel_3x3 = np.ones((3, 3), np.float32) / 9                            #3x3 kernel creating
blurred = cv2.filter2D(image, -1, kernel_3x3)
plt.imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
plt.title('Blurred 3x3')
plt.show()
cv2.imwrite('3x3_blurred.png', blurred)


# In[8]:


kernel_7x7 = np.ones((7, 7), np.float32) / 49                            #7x7 kernel creating
blurred = cv2.filter2D(image, -1, kernel_7x7)
plt.imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
plt.title('Blurred 7x7')
plt.show()
cv2.imwrite('7x7_blurred.png', blurred)


# In[11]:


kernel_sizes = [3, 5, 7]  

for kernel_size in kernel_sizes:
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
    blurred = cv2.filter2D(image, -1, kernel)
    
    plt.imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
    plt.title(f'Blurred {kernel_size}x{kernel_size}')
    plt.show()


# ## Averaging :

# In this process, each pixel is replaced by the average value of its neighboring pixels, creating a simple and uniform smoothing effect.

# In[12]:


blur = cv2.blur(image, (3, 3))
plt.imshow(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB))
plt.title('Average Blur')
plt.show()
cv2.imwrite('average_blur.png', blur)


# ## Gaussian Blurring :

# In this process, similar to averaging, but the neighborhood pixel values are weighted based on a Gaussian distribution. This results in a smoother image while preserving more of the edges and fine details.

# In[14]:


gaussian_blur = cv2.GaussianBlur(image, (7, 7), 0)
plt.imshow(cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2RGB))
plt.title('Gaussian Blur')
plt.show()
cv2.imwrite("gaussian_blur.png", gaussian_blur)


# # Median Blurring :

# In this process, the central pixel is replaced by the median value of its surrounding pixels. This method is effective at reducing salt-and-pepper noise while preserving edges.

# In[15]:


median_blur = cv2.medianBlur(image, 5)
plt.imshow(cv2.cvtColor(median_blur, cv2.COLOR_BGR2RGB))
plt.title('Median Blur')
plt.show()
cv2.imwrite('median_blur.png', median_blur)


# # Bilateral Blurring :

# In this process, both the spatial distance and the intensity difference between pixels. It preserves edges while effectively reducing noise.

# In[17]:


bilateral_blur = cv2.bilateralFilter(image, 9, 75, 75)
plt.imshow(cv2.cvtColor(bilateral_blur, cv2.COLOR_BGR2RGB))
plt.title('Bilateal Blur')
plt.show()
cv2.imwrite('bilateral_blur.png', bilateral_blur)


# In[18]:


row, col = 2, 2
fig, axs = plt.subplots(row, col, figsize=(10, 5))
fig.tight_layout()

blur_filters = [
    (cv2.blur, 'Averaging', 'Average Blurring.png', (3, 3)),
    (cv2.GaussianBlur, 'Gaussian', 'Gaussian Blurring.png', (7, 7), 0),
    (cv2.medianBlur, 'Median', 'Median Blurring.png', 5),
    (cv2.bilateralFilter, 'Bilateral', 'Bilateral Blurring.png', 9, 75, 75)
]

for i, (filter_func, title, filename, *params) in enumerate(blur_filters):
    filtered_image = filter_func(image, *params)
    axs[i // col][i % col].imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
    axs[i // col][i % col].set_title(title)
    cv2.imwrite(filename, filtered_image)

plt.show()


# In[ ]:




