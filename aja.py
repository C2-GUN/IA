# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 10:02:33 2019

@author: CDEC
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 24 09:46:47 2019

@author: CDEC
"""

import cv2
import numpy as np
kernel = np.ones((5,5),np.uint8)

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged





for i in range(1,21):     
    
    path = 'D:\Documents\OPENCV\Placas\Placa ('+str(i)+").jpg"
    img = cv2.imread(path)
    
    numero = 0       
    
    
     
    
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lower_blue = np.array([0, 0, 0])
    upper_blue = np.array([179, 255, 115])
    
    mask = cv2.inRange(hsv, lower_blue, upper_blue) 
    result = cv2.bitwise_and(img, img, mask=mask)
    
    gray2 = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    
    thresh = cv2.adaptiveThreshold(gray2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 4)
    
    _, thresh2 = cv2.threshold(gray2, 110, 255, cv2.THRESH_BINARY)
    
    img2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(img, contours, -1, (0,255,0), 1, cv2.LINE_AA)      

    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        (x, y, w, h) = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h               
        rect_area = w*h
        extension = float(area)/rect_area
        if(area > 500 ):
            (x, y, w, h) = cv2.boundingRect(cnt)
            #cv2.drawContours(img,[cnt],0,(0,0,255),-1,4)
            cv2.rectangle(img, (x,y), (x+w,y+h), (255, 0, 0), 2)  
            
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
    cv2.imshow('Placa', result)
    cv2.waitKey()
    
    
    cv2.namedWindow('img2', cv2.WINDOW_NORMAL)
    cv2.imshow('img2', thresh)
    cv2.waitKey()
 
    
    '''cv2.namedWindow('img4', cv2.WINDOW_NORMAL)
    cv2.imshow('img4', edge)
    cv2.waitKey()'''
    

    
    

    
    cv2.destroyAllWindows()