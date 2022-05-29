#!/usr/bin/env python
# coding: utf-8

# In[34]:


import cv2
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

#Loading the image
img = cv.imread('DSC_0117.jpg')
RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(RGB_img),plt.colorbar(),plt.show()

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
     
    # Threshold of brown in HSV space
lower_blue = np.array([12, 100, 20])
upper_blue = np.array([28, 255, 200])
 
    # preparing the mask to overlay
mask = cv2.inRange(hsv, lower_blue, upper_blue)
     
result = cv2.bitwise_and(img, img, mask = mask)
 
plt.imshow(result),plt.colorbar(),plt.show()
 


# In[ ]:




