# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 02:11:45 2019

@author: CDEC
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt



from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature
from skimage.feature import haar_like_feature
from skimage.transform import integral_image
import imutils


feature_types = ['type-2-x', 'type-2-y', 'type-3-x', 'type-3-y','type-4']

path = 'D:\\Documents\\OPENCV\\TRAINING\\2.jpg'
path2 = '\3.jpg'

def extract_feature_image(img, feature_type, feature_coord=None):
    """Extract the haar feature for the current image"""
    ii = integral_image(img)
    return haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1],
                             feature_type=feature_type,
                             feature_coord=feature_coord)

def sliding_window_(image, stepSize, windowSize):	
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):			
            window = image[x:x + windowSize, y:y + windowSize]
            cv2.namedWindow('Digito1', cv2.WINDOW_NORMAL)
            cv2.imshow('Digito1', window)
            clone = image.copy()
            cv2.rectangle(image, (x, y), (x + windowSize, y + windowSize), (0, 255, 0), 0)
            cv2.namedWindow('Digito', cv2.WINDOW_NORMAL)
            cv2.imshow('Digito', clone)
            cv2.waitKey()
            cv2.destroyAllWindows()
            
    return window     

def pyramid(image, scale=1.5, minSize=(8, 8)):
    flag = True
    while flag == True:

        print('IN')
        
        cv2.namedWindow('Digito', cv2.WINDOW_NORMAL)
        cv2.imshow('Digito', image)
        print(image.shape)
        cv2.waitKey()
        cv2.destroyAllWindows()

        
        if image.shape[0] <= minSize[1] or image.shape[1] <= minSize[0]:
            flag = False
        else: 
            W = image.shape[0] - 8
            H = image.shape[1] - 8
            dim = (W,H)
            image = cv2.resize(image, dim)
		
        print(flag)
    return image
    
            



print(path)
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


#X = delayed(extract_feature_image(img, feature_types)
        #for imgs in img)
#print(X)
    
 
#x= extract_feature_image(img,'type-4', feature_coord=None)






sliding_window_(img, stepSize = 8, windowSize = 8)
        


pyramid(img, scale=1.5)





#img = cv2.imread('D:\Documents\OPENCV\DB_PLATES\carro (1).jpg')

