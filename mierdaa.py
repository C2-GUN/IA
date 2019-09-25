# -*- coding: utf-8 -*-
"""
Created on Sun May 12 04:14:14 2019

@author: CDEC
"""

import cv2
import numpy as np


    
  
        
      

def extract1(path):
    
    image = cv2.imread(path)
    plate_flag = False
    Plate = []
    
    numero = 0
    
    [fil,col,chn] = image.shape
 
    XX = col/4
    YY = fil/4    
    
    img = image[int(YY):2100, int(XX)-80:2100]
    
    rgB=np.matrix(img[:,:,0])
    rGb=np.matrix(img[:,:,1])
    Rgb=np.matrix(img[:,:,2])
    
    IImage = cv2.absdiff(rGb,rgB)
    
    I=IImage
    
    
    t, thresh = cv2.threshold(I, 45, 255, cv2.THRESH_BINARY)
    
    
    imageContours, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    ##cv2.drawContours(img, contours, -1, (0,255,0), 4, cv2.LINE_AA)
    
    
    
    
    for cnt in contours:
        
        area = cv2.contourArea(cnt)
        (x, y, w, h) = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h
               
        rect_area = w*h
        extension = float(area)/rect_area
        
        if (aspect_ratio >= 2.0 and aspect_ratio <= 3.5):
            if(extension >= 0.3):
                if(area >= 7000 and area <= 135000):
                    (x, y, w, h) = cv2.boundingRect(cnt)
                    ##cv2.drawContours(img,[cnt],0,(0,0,255),-1)
                    cv2.rectangle(img, (x,y), (x+w,y+h), 0)
                    hull = cv2.convexHull(cnt)
                    hull_area = cv2.contourArea(hull) 
                    solidez = float(area)/hull_area
                    Plate = img[y:y+h,x:x+w]
                    
                    plate_flag = True
                    
                        
                    content = True
                    
                    numero += 1                                   
                       
                    
                    print (i)
                    print (numero)
                    print ("im here")
                    
                    break
                    
                    '''cv2.namedWindow('Vehiculo', cv2.WINDOW_NORMAL)
                    cv2.imshow('Vehiculo', image)
                    cv2.waitKey()
        
                    cv2.namedWindow('Placa', cv2.WINDOW_NORMAL)
                    cv2.imshow('Placa', Plate)
                    cv2.waitKey()
                    cv2.destroyAllWindows()
                    
                    Plate = []'''
                    
                    print("block 1" '\n')
                    
                    
                    '''print("X: ", x ,'\n' 
                                  "Y: ", y, '\n'
                                  "W: ", w , '\n'
                                  "H: ", h, '\n'
                                  "Y+H: ",y+h, '\n'
                                  'X+W: ',x+w, '\n'                      
                                  "area: ",area ,'\n'
                                  'aspect_ratio...1: ',aspect_ratio, '\n'
                                  'Extension: ',extension, '\n'
                                  "Solidez: ",solidez ,'\n'
                                  "-------------------------------------------------------",'\n')'''
    print (len(Plate))                
    return Plate
                    
            
    
    

                    


                

for i in range(24,27):     
    
    path = 'D:\Documents\OPENCV\DB_PLATES\carro ('+str(i)+").jpg"
    content = False
    img = cv2.imread(path)
    
    print (extract1(path))
    
    cv2.namedWindow('Vehiculo', cv2.WINDOW_NORMAL)
    cv2.imshow('Vehiculo', img)
    cv2.waitKey()
    
    
    if(len(extract1(path))==0):
        print("excuse muaaaa")
       
    else:            
        cv2.namedWindow('Placa', cv2.WINDOW_NORMAL)
        cv2.imshow('Placa', extract1(path))
        cv2.waitKey()
        cv2.destroyAllWindows()
 
    
