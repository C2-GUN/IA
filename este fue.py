# -*- coding: utf-8 -*-
"""
Created on Fri May 24 09:46:47 2019

@author: CDEC
"""

import cv2
import numpy as np
kernel = np.ones((5,5),np.uint8)


for i in range(1,21):     
    
    path = 'D:\Documents\OPENCV\Placas\Placa ('+str(i)+").jpg"
    img = cv2.imread(path)
    
    numero = 0    
    
    caracteres = []
   
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 4)
    
    edgess = cv2.dilate(thresh, None,iterations=1)
    
    yu = cv2.GaussianBlur(thresh, (5, 5), 0)
    edge = cv2.Canny(yu, 180, 260)
    

    
    imageContours, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(img, contours, -1, (0,255,0), 1, cv2.LINE_AA)      

    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        (x, y, w, h) = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h               
        rect_area = w*h
        extension = float(area)/rect_area
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
                    
        equi_diametro = np.sqrt(4*area/np.pi)
        
        mask = np.zeros(gray.shape,np.uint8)
        cv2.drawContours(mask,[cnt],0,255,-1)
        pixelpoints = np.transpose(np.nonzero(mask))
        pixelpoints = cv2.findNonZero(mask)
        
        if(aspect_ratio > 0.1 and aspect_ratio < 1 and area >= 650 and equi_diametro >= 32 and extension >= 0.1):
            (x, y, w, h) = cv2.boundingRect(cnt)
            #cv2.drawContours(img,[cnt],0,(0,0,255),-1,4)
            cv2.rectangle(img, (x,y), (x+w,y+h), (255, 0, 0), 2)
            solidez = float(area)/hull_area
            
            
            Plate = img[y:y+h,x:x+w]               
            caracteres.append(img[y:y+h,x:x+w])
            
            
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
                  'Solidez: ',solidez, '\n'
                  'Diametro: ',equi_diametro, '\n'
                  'PIXEL: ',len(pixelpoints), '\n'                   
                  "-------------------------------------------------------",'\n')
                    
    
    print ('caracteres: ',len(caracteres), '\n')
    
    
    
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', img)
    cv2.waitKey()

 
    
    '''cv2.namedWindow('img4', cv2.WINDOW_NORMAL)
    cv2.imshow('img4', edge)
    cv2.waitKey()'''
    

    
    

    
    cv2.destroyAllWindows()