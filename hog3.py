# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 09:36:40 2019

@author: CDEC
"""

import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure
import cv2

sourcer_params = {
  'color_model': 'hsv',                # hls, hsv, yuv, ycrcb
  'number_of_orientations': 12,        # 6 - 12
  'pixels_per_cell': 8,               # 8, 16
  'cells_per_block': 2,                # 1, 2
  'transform_sqrt': True
}


image = data.astronaut()

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print('tamaño: ', image.shape)

fd, hog_image = hog(image, orientations=12, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualise=True, feature_vector = True, transform_sqrt=True)


features, hog_img = hog(image, 
                        orientations = sourcer_params['number_of_orientations'], 
                        pixels_per_cell = (sourcer_params['pixels_per_cell'], sourcer_params['pixels_per_cell']),
                        cells_per_block = (sourcer_params['cells_per_block'], sourcer_params['cells_per_block']), 
                        transform_sqrt = sourcer_params['transform_sqrt'], 
                        visualise = True, 
                        feature_vector = True)

print(fd.shape[0])
print(features.shape[0])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()

cv2.namedWindow('1', cv2.WINDOW_NORMAL)
cv2.imshow('1', hog_image)
cv2.waitKey()
cv2.destroyAllWindows()
        