# -*- coding: utf-8 -*-
"""
Created on Thu May 23 11:08:00 2019

@author: CDEC
"""

import cv2
import numpy as np
kernel = np.ones((3,3),np.uint8)

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
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 105, 8)
    
    cierre = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    apertura = cv2.morphologyEx(cierre, cv2.MORPH_OPEN, kernel)
    
    gradiente = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)
    gradiente1 = cv2.GaussianBlur(gradiente, (5, 5), 0)
    
    yu = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(yu, 180, 255)
    edge = cv2.Canny(thresh, 180, 260)
    
    auto = auto_canny(blur)
    edgess = cv2.dilate(auto, None,iterations=1)
    
    t, thresh1 = cv2.threshold(blur, 110, 255, cv2.THRESH_BINARY_INV)
    
    t, threshnew = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
    
    transformacion = cv2.dilate(thresh,kernel,iterations = 1) - cv2.erode(thresh,kernel,iterations = 1)
    
    imageContours, contours, hierarchy = cv2.findContours(edgess, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(img, contours, -1, (0,255,0), 1, cv2.LINE_AA)       

    
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt,0.02*cv2.arcLength(cnt,True),True)
        x,y,w,h = cv2.boundingRect(cnt)
        diag = h/w
        #if(diag >= 0.3 and diag <= 0.48):        
        area = cv2.contourArea(cnt)
        ar = float(w)/float(h)
        aspect_ratio = float(w)/h
        #cv2.drawContours(img,[cnt],0,(0,0,255),-1)
        if (area >= 500):
            if(aspect_ratio >= 0.3 and aspect_ratio <= 0.7):
                cv2.rectangle(img, (x,y), (x+w,y+h), (255, 0, 255), 3)            
                rect_area = w*h
                extension = float(area)/rect_area
                        
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
                              'APROX: ',approx, '\n'
                              "-------------------------------------------------------",'\n')
                    
    
    
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', img)
    cv2.waitKey()
    
    cv2.namedWindow('Placa', cv2.WINDOW_NORMAL)
    cv2.imshow('Placa', edgess)
    cv2.waitKey()
    
    
    cv2.namedWindow('imge', cv2.WINDOW_NORMAL)
    cv2.imshow('imge', edge)
    cv2.waitKey()
    
    '''cv2.namedWindow('imge1', cv2.WINDOW_NORMAL)
    cv2.imshow('imge1', apertura)
    cv2.waitKey()
    
    cv2.namedWindow('img2e1', cv2.WINDOW_NORMAL)
    cv2.imshow('img2e1', transformacion)
    cv2.waitKey()
    
    cv2.namedWindow('img2e11', cv2.WINDOW_NORMAL)
    cv2.imshow('img2e11', gradiente1)
    cv2.waitKey()'''
 
 
    

    
    

    
    cv2.destroyAllWindows()