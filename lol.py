# -*- coding: utf-8 -*-
"""
Created on Thu May  9 23:39:04 2019

@author: CDEC
"""
import cv2
import numpy as np
#import math
#import argparse





for i in range(59,63):     
    
    path = 'D:\Documents\OPENCV\DB_PLATES\carro ('+str(i)+").jpg"
    content = False
    image = cv2.imread(path)
    
    numero = 0
    
    [fil,col,chn] = image.shape
 
    XX = col/4
    YY = fil/4    
    
    img = image[int(YY):2800, int(XX)-80:2100]
    
    rgB=np.matrix(img[:,:,0])
    rGb=np.matrix(img[:,:,1])
    Rgb=np.matrix(img[:,:,2])
    
    IImage = cv2.absdiff(rGb,rgB)
    
    I=IImage
    
    
    t, thresh = cv2.threshold(I, 45, 255, cv2.THRESH_BINARY)
    
    
    imageContours, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(img, contours, -1, (0,255,0), 4, cv2.LINE_AA)
    
    canny = cv2.Canny(thresh, 1, 0.4, 0.2)
    
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 35000 and area < 100000:
            (x, y, w, h) = cv2.boundingRect(cnt)
            cv2.drawContours(img,[cnt],0,(0,0,255),-1)
            cv2.rectangle(img, (x,y), (x+w,y+h), (255, 0, 255), 7)
            aspect_ratio = float(w)/h
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidez = float(area)/hull_area
            rect_area = w*h
            extension = float(area)/rect_area
            
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
                      '1). aspect_ratio: ',aspect_ratio, '\n'
                      '1). Extension: ',extension, '\n'
                      "1). Solidez: ",solidez ,'\n'
                      "---------------------------------------------------------------------",'\n')
        else:
            (x, y, w, h) = cv2.boundingRect(cnt)
            aspect_ratio = float(w)/h
            area = cv2.contourArea(cnt)
            if(aspect_ratio >= 2.0):
                if(w>=200):
                    cv2.drawContours(img,[cnt],0,(0,0,255),-1)
                    cv2.rectangle(img, (x,y), (x+w,y+h), (255, 0, 255), 7)
                    hull = cv2.convexHull(cnt)
                    hull_area = cv2.contourArea(hull)
                    solidez = float(area)/hull_area
                    rect_area = w*h
                    extension = float(area)/rect_area
                    
                    content = True
                    numero += 1
                    print (i)
                    print (numero, '\n')
                    print("W: ", w , '\n'
                              "H: ", h, '\n'
                              "Y+H: ",y+h, '\n'
                              'X+W: ',x+w, '\n'
                              "area: ",area ,'\n'
                              'aspect_ratio: ',aspect_ratio, '\n'
                              'Extension: ',extension, '\n'
                              "Solidez: ",solidez ,'\n'
                              "---------------------------------------------------------------------",'\n')
            
    '''if(content == False):
        print("PERROO PAILAAA")
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt,0.02*cv2.arcLength(cnt,True),True)
            if len(approx)==4:
                x,y,w,h = cv2.boundingRect(cnt)
                diag = h/w
                #if(diag >= 0.3 and diag <= 0.48):        
                area = cv2.contourArea(cnt)
                ar = float(w)/float(h)
                cv2.drawContours(img,[cnt],0,(0,0,255),-1)
                cv2.rectangle(img, (x,y), (x+w,y+h), (255, 0, 255), 7)
                
                    
                print("X: ", x ,'\n' 
                      "Y: ", y, '\n'
                      "W: ", w , '\n'
                      "H: ", h, '\n'
                      "Y+H: ",y+h, '\n'
                      'X+W: ',x+w, '\n'
                      'Division: ',h/w, '\n'
                      "area: ",area ,'\n')            
                

    

    cv2.namedWindow('value3', cv2.WINDOW_NORMAL)
    cv2.imshow('value3', thresh)
    cv2.waitKey()   
    
    cv2.namedWindow('value4', cv2.WINDOW_NORMAL)
    cv2.imshow('value4', canny)
    cv2.waitKey()
    
    cv2.namedWindow('value1', cv2.WINDOW_NORMAL)
    cv2.imshow('value1', image)
    cv2.waitKey()''' 
    
    cv2.namedWindow('value', cv2.WINDOW_NORMAL)
    cv2.imshow('value', img)
    cv2.waitKey()  
    

    
    

    cv2.destroyAllWindows()