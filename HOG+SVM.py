# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 00:22:29 2019

@author: CDEC
"""

import cv2
import numpy as np
from sklearn.externals import joblib
from skimage.feature import hog


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


def Hoog(img, sourcer_params):
         
    features, hog_img = hog(img, 
                        orientations = sourcer_params['number_of_orientations'], 
                        pixels_per_cell = (sourcer_params['pixels_per_cell'], sourcer_params['pixels_per_cell']),
                        cells_per_block = (sourcer_params['cells_per_block'], sourcer_params['cells_per_block']), 
                        transform_sqrt = sourcer_params['transform_sqrt'], 
                        visualise = True, 
                        feature_vector = True,
                        block_norm='L2-Hys')
    

    return features, hog_img


svm = joblib.load('D:\Documents\OPENCV\MODELS\svm_model.pkl')

img = cv2.imread('D:\\Documents\\OPENCV\\TRAINING\\1.jpg')

imgX = change_color(img, sourcer_params)        
(features, hog_img) = Hoog(imgX, sourcer_params)

nbr = svm.predict(np.array([features], 'float64'))
print(nbr[0])