# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 18:57:21 2019

@author: CDEC
"""
from sklearn.externals import joblib
from skimage import feature
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import cv2


imagepath = 'D:\\Documents\\OPENCV\\TRAINING\\1.jpg'
image = cv2.imread(imagepath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.style.use("ggplot")
(fig, ax) = plt.subplots()
fig.suptitle("Local Binary Patterns")
plt.ylabel("% of Pixels")
plt.xlabel("LBP pixel bucket")

# plot a histogram of the LBP features and show it

# displaying default to make cool image
features = feature.local_binary_pattern(gray, 10, 10, method="default") # method="uniform")

cv2.namedWindow('Digito grey', cv2.WINDOW_NORMAL)
cv2.imshow('Digito grey', features.astype("uint8"))
cv2.waitKey()
cv2.destroyAllWindows()

# Save figure of lbp_image


ax.hist(features.ravel(), normed=True, bins=20, range=(0, 256))
ax.set_xlim([0, 256])
ax.set_ylim([0, 0.030])
# save figure

plt.show()

cv2.destroyAllWindows()