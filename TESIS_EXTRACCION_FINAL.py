# -*- coding: utf-8 -*-
"""
Created on Thu May 16 07:33:25 2019

@author: CDEC
"""

import cv2
import numpy as np


def extract(path):
    
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
    
    
    IImage = cv2.absdiff(rGb,rgB)
    
    I=IImage
    
    #I = cv2.GaussianBlur(I, (5, 5), 0)
    
    
    t, thresh = cv2.threshold(I, 45, 255, cv2.THRESH_BINARY)
    
    
    imageContours, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    ##cv2.drawContours(img, contours, -1, (0,255,0), 4, cv2.LINE_AA)
    
    
    
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        (x, y, w, h) = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h
               
        rect_area = w*h
        extension = float(area)/rect_area
        
        if (aspect_ratio >= 2.0 and aspect_ratio <= 3.4):
            if(extension >= 0.3):
                if(area >= 9000 and area <= 135000):
                    (x, y, w, h) = cv2.boundingRect(cnt)
                    ##cv2.drawContours(img,[cnt],0,(0,0,255),-1)
                    cv2.rectangle(img, (x,y), (x+w,y+h), 0)
                    
                    
                    Plate = img[y:y+h,x:x+w]
                    
                    plate_flag = True           
                        

                    numero += 1                                   
                       
                    
                    print (i)
                    print (numero)
                    
                    ''''cv2.namedWindow('Vehiculo', cv2.WINDOW_NORMAL)
                    cv2.imshow('Vehiculo', image)
                    cv2.waitKey()
        
                    cv2.namedWindow('Placa', cv2.WINDOW_NORMAL)
                    cv2.imshow('Placa', Plate)
                    cv2.waitKey()
                    cv2.destroyAllWindows()     '''              
                    
                    
                    print("block 1" '\n')
                    break
                     
                    
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
    
    
    

    if(plate_flag == False):
        
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
    
        #cv2.drawContours(img, contours, -1, (0,255,0), 4, cv2.LINE_AA)
    
    
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
                #cv2.drawContours(img,[cnt],0,(0,0,255),-1)                
                
                
                
                Plate = img[y:y+h,x:x+w]
                plate_flag = True
                
                    
                
                numero += 1                                   
                                   
                print (i)
                print (numero)
                '''cv2.namedWindow('Vehiculo', cv2.WINDOW_NORMAL)
                cv2.imshow('Vehiculo', image)
                cv2.waitKey()
                
                cv2.namedWindow('Placa', cv2.WINDOW_NORMAL)
                cv2.imshow('Placa', Plate)
                cv2.waitKey()
                cv2.destroyAllWindows()'''
                
                print("block 2" '\n')
                
                                
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
                
            
                
                
                
                   

    if(plate_flag == False):
       print('\n' "**** TRY ANOTHER IMAGE ****")

    return Plate      


folder = 'D:\Documents\OPENCV\Placas'     
for i in range(35,36):     
    
    path = 'D:\Documents\OPENCV\DB_PLATES\carro ('+str(i)+").jpg"
    car = cv2.imread(path)
    
    cv2.namedWindow('Vehiculo', cv2.WINDOW_NORMAL)
    cv2.imshow('Vehiculo', car)
    cv2.waitKey()
    
    
    
    if(len(extract(path))==0):
        print("ERROR")
       
    else:
        placa = extract(path) 
        
        name = "Placa" + path[35:]
        
        cv2.imwrite('%s/%s' % (folder,name), placa)
        
        print(folder)
        print(name)
        
        im = cv2.cvtColor(placa, cv2.COLOR_BGR2GRAY)
        im = cv2.GaussianBlur(im, (5, 5), 0)
        
        t, thresh = cv2.threshold(im, 100, 255, cv2.THRESH_BINARY_INV)
        imageContours, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rects = [cv2.boundingRect(ctr) for ctr in contours]
        
        for rect in rects:
            
            cv2.rectangle(placa, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
        
        cv2.namedWindow('Placa', cv2.WINDOW_NORMAL)
        cv2.imshow('Placa', placa)
        cv2.waitKey()
        cv2.destroyAllWindows()
