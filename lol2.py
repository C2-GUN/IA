# -*- coding: utf-8 -*-
"""
Created on Fri May 10 01:19:16 2019

@author: CDEC
"""

import cv2

for i in range(11,12):     
    
    path = 'D:\Documents\OPENCV\DB_PLATES\carro ('+str(i)+").jpg"
    image1 = cv2.imread(path)
    image = image1
    
    b,g,r=cv2.split(image)
    image = cv2.merge(g,r)
    
    cv2.imshow(image)
    cv2.waitKey(0)
    cv2.destroyAllWindows