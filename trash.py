# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 00:55:01 2019

@author: CDEC
"""


import cv2
#import numpy as np
#import math
#import argparse




for i in range(19,41):     
    
    path = 'D:\Documents\OPENCV\DB_PLATES\carro ('+str(i)+").jpg"
    image1 = cv2.imread(path)
    image = image1
    
    img = image[1000:2000, 1000:2000]
    
    
    hsv = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    topHat = cv2.morphologyEx(value, cv2.MORPH_TOPHAT, kernel)
    blackHat = cv2.morphologyEx(value, cv2.MORPH_BLACKHAT, kernel)
    
    add = cv2.add(value, topHat)
    subtract = cv2.subtract(add, blackHat)
    blur = cv2.GaussianBlur(subtract, (5, 5), 0)
    
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)
    
    imageContours, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(image1, contours, -1, (0,255,0), 4, cv2.LINE_AA)
    
    
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt,0.02*cv2.arcLength(cnt,True),True)
        if len(approx)==4:
            x,y,w,h = cv2.boundingRect(cnt)
            diag = h/w
            #if(diag >= 0.3 and diag <= 0.48):        
            area = cv2.contourArea(cnt)
            ar = float(w)/float(h)
            cv2.drawContours(image1,[cnt],0,(0,0,255),-1)
            cv2.rectangle(image1, (x,y), (x+w,y+h), (255, 0, 255), 7)
            type(x)
                
            print("X: ", x ,'\n' 
                  "Y: ", y, '\n'
                  "W: ", w , '\n'
                  "H: ", h, '\n'
                  "Y+H: ",y+h, '\n'
                  'X+W: ',x+w, '\n'
                  'Division: ',h/w, '\n'
                  "area: ",area ,'\n') 
    
    
    cv2.namedWindow('value', cv2.WINDOW_NORMAL)
    cv2.imshow('value', image1)
    cv2.waitKey()
    cv2.destroyAllWindows()







