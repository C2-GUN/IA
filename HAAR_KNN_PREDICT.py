# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 01:29:26 2019

@author: CDEC
"""

import cv2
import numpy as np
kernel = np.ones((5,5),np.uint8)
import matplotlib.pyplot as plt
import glob
from pathlib import Path
from sklearn.model_selection import  train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from sklearn import metrics
from skimage.feature import haar_like_feature
from skimage.transform import integral_image
from timeit import default_timer as timer

start = timer()

folder = 'D:\Documents\OPENCV\TRAINING'
width = 40
height = 40
dim = (width, height)
label_list = []
features_list = []
features_windows = []
window_feature_IN_piramid = []
window = []
feature_types = ['type-4']
windows = []


sourcer_params = {
  'color_model': 'grey',                # hls, hsv, yuv, ycrcb
  'bounding_box_size': 64,             #
  'number_of_orientations': 12,        # 6 - 12
  'pixels_per_cell': 8,               # 8, 16
  'cells_per_block': 2,                # 1, 2
  'transform_sqrt': True
}


def change_color(img, sourcer_params):
    
    
    if sourcer_params['color_model'] == "hsv": 
      img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif sourcer_params['color_model'] == "hls":
      img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    elif sourcer_params['color_model'] == "yuv":
      img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    elif sourcer_params['color_model'] == "ycrcb":
      img = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
    elif sourcer_params['color_model'] == "grey":
      img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else: 
      raise Exception('ERROR:', 'No se puede cambiar de color')
    
    '''hogA_img = img[:, :, 0]
    hogB_img = img[:, :, 1]
    hogC_img = img[:, :, 2]'''
    
    return img#, hogA_img, hogB_img, hogC_img


 

knn = joblib.load('D:\Documents\OPENCV\MODELS\HAAR_KNN_MODEL_0.9291044776119403.pkl')

img = cv2.imread('D:\\Documents\\OPENCV\\TRAINING\\2.jpg')
imgGREY = change_color(img, sourcer_params)
imgX = integral_image(imgGREY)        
features = haar_like_feature(imgX, 0, 0, 40, 40, feature_types)

nbr = knn.predict(np.array([features]))
print(nbr[0])
end = timer()
print("{0:.3f}".format(end - start)+' seconds') # Time in seconds