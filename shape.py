# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 10:59:43 2019

@author: CDEC
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt



from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature
from skimage.feature import haar_like_feature
from skimage.transform import integral_image
from dask import delayed
feature_types = ['type-2-x', 'type-2-y', 'type-3-x', 'type-3-y','type-4']

path = 'D:\\Documents\\OPENCV\\TRAINING\\4.jpg'
path2 = '\3.jpg'

def extract_feature_image(img, feature_type, feature_coord=None):
    """Extract the haar feature for the current image"""
    ii = integral_image(img)
    return haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1],
                             feature_type=feature_type,
                             feature_coord=feature_coord)

TOTAL = path + path2
print(path)
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
feat_coord, feat_type = haar_like_feature_coord(2, 2, feature_types)


features = draw_haar_like_feature(img, 2, 3, 39, 39, feat_coord)


#X = delayed(extract_feature_image(img, feature_types)
        #for imgs in img)
#print(X)
    
 
#x= extract_feature_image(img,'type-4', feature_coord=None)


img2 = integral_image(img)
feature = haar_like_feature(img, 0, 0, 7, 7, feature_types)
print(len(feature))
print(feature)



#img = cv2.imread('D:\Documents\OPENCV\DB_PLATES\carro (1).jpg')

cv2.namedWindow('Digito', cv2.WINDOW_NORMAL)
cv2.imshow('Digito', features)
cv2.waitKey()
cv2.destroyAllWindows()
