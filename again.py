# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:38:46 2019

@author: CDEC
"""

import cv2
import numpy as np
kernel = np.ones((4,4),np.uint8)



for i in range(1,11):     
    
    path = 'D:\Documents\OPENCV\Placas\Placa ('+str(i)+").jpg"
    img = cv2.imread(path)
    
    numero = 0 
    
    rgB=np.matrix(img[:,:,0])
    rGb=np.matrix(img[:,:,1])
    
    
    IImage = cv2.absdiff(rGb,rgB)
    
    I=IImage
    
    t, thresh2 = cv2.threshold(I, 10, 255, cv2.THRESH_BINARY)

    
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    thresh = cv2.adaptiveThreshold(I, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 125, 8)
    
    cierre = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    yu = cv2.GaussianBlur(thresh, (5, 5), 0)
    edged = cv2.Canny(yu, 75, 200)
    
    t, thresh1 = cv2.threshold(blur, 110, 255, cv2.THRESH_BINARY_INV)
    
    t, threshnew = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
    
    transformacion = cv2.dilate(thresh,kernel,iterations = 1) - cv2.erode(thresh,kernel,iterations = 1)
    
    imageContours, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(img, contours, -1, (0,255,0), 3, cv2.LINE_AA)
    

        

    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        (x, y, w, h) = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h               
        rect_area = w*h
        extension = float(area)/rect_area
        if(area > 700 ):
            (x, y, w, h) = cv2.boundingRect(cnt)
            #cv2.drawContours(img,[cnt],0,(0,0,255),-1,4)
            cv2.rectangle(img, (x,y), (x+w,y+h), 0)
            
            numero += 1                                   
            print (i)
            print (numero ,'\n')
            print("X: ", x ,'\n' 
                  "Y: ", y, '\n'
                  "W: ", w , '\n'
                  "H: ", h, '\n'
                  "Y+H: ",y+h, '\n'
                  'X+W: ',x+w, '\n'                      
                  "area: ",area ,'\n'
                  'aspect_ratio...1: ',aspect_ratio, '\n'
                  'Extension: ',extension, '\n'
                  "-------------------------------------------------------",'\n')
                    
    
    
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', img)
    cv2.waitKey()
    
    cv2.namedWindow('Placa', cv2.WINDOW_NORMAL)
    cv2.imshow('Placa', I)
    cv2.waitKey()
    
    
    cv2.namedWindow('img1', cv2.WINDOW_NORMAL)
    cv2.imshow('img1', thresh)
    cv2.waitKey()
 
    
    '''cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', edged)
    cv2.waitKey()'''
    

    
    

    
    cv2.destroyAllWindows()