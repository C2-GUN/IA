# -*- coding: utf-8 -*-
"""
Created on Sat May 11 16:14:39 2019

@author: CDEC
"""

import cv2
#import numpy as np
#import math
#import argparse




for i in range(75,76):     
    
    path = 'D:\Documents\OPENCV\DB_PLATES\carro ('+str(i)+").jpg"
    image1 = cv2.imread(path)
    image = image1
    
    [fil,col,chn] = image.shape
 
    XX = col/4
    YY = fil/4    
    
    img = image[int(YY):2100, int(XX)-80:2100]
    
    numero = 0    
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    topHat = cv2.morphologyEx(value, cv2.MORPH_TOPHAT, kernel)
    blackHat = cv2.morphologyEx(value, cv2.MORPH_BLACKHAT, kernel)
    
    add = cv2.add(value, topHat)
    subtract = cv2.subtract(add, blackHat)
    blur = cv2.GaussianBlur(subtract, (5, 5), 0)
    
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)
    
    imageContours, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(img, contours, -1, (0,255,0), 4, cv2.LINE_AA)
    
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        (x, y, w, h) = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h
               
        rect_area = w*h
        extension = float(area)/rect_area
        
        ##if (aspect_ratio >= 2.0 and aspect_ratio <= 3.0):
            ##if(extension >= 0.3):
        if(area >= 40000 and area <= 150000):
            (x, y, w, h) = cv2.boundingRect(cnt)
            cv2.drawContours(img,[cnt],0,(0,0,255),-1)
            cv2.rectangle(img, (x,y), (x+w,y+h), (255, 0, 255), 7)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull) 
            solidez = float(area)/hull_area
            
                
            content = True
            numero += 1                                   
                               
            print (i)
            print (numero, '\n') 
                            
            print("X: ", x ,'\n' 
                  "Y: ", y, '\n'
                  "W: ", w , '\n'
                  "H: ", h, '\n'
                  "Y+H: ",y+h, '\n'
                  'X+W: ',x+w, '\n'                      
                  "area: ",area ,'\n'
                  'aspect_ratio...1: ',aspect_ratio, '\n'
                  'Extension: ',extension, '\n'
                  "Solidez: ",solidez ,'\n'
                  "-------------------------------------------------------",'\n')
                    
        '''else:
            cv2.drawContours(img,[cnt],0,(255,255,0),-1)
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,255), 7)'''
        

    
    cv2.namedWindow('value', cv2.WINDOW_NORMAL)
    cv2.imshow('value', image1)
    cv2.waitKey()
    cv2.destroyAllWindows()

    
    cv2.namedWindow('value', cv2.WINDOW_NORMAL)
    cv2.imshow('value', img)
    cv2.waitKey()
    
    
    cv2.destroyAllWindows()