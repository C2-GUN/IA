# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 13:28:02 2019

@author: CDEC
"""

import cv2
import numpy as np
kernel = np.ones((5,5),np.uint8)
from featuresourcer import FeatureSourcer
from helpers import show_images, convert
import matplotlib.pyplot as plt



sourcer_params = {
  'color_model': 'hsv',                # hls, hsv, yuv, ycrcb
  'bounding_box_size': 64,             #
  'number_of_orientations': 12,        # 6 - 12
  'pixels_per_cell': 8,               # 8, 16
  'cells_per_block': 2,                # 1, 2
  'do_transform_sqrt': True
}

winSize = (20,20)
blockSize = (10,10)
blockStride = (5,5)
cellSize = (10,10)
nbins = 9
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
useSignedGradients = True


 

def characters(path):
    img = cv2.imread(path)
    
    numero = 0    
    
    caracteres = []
   
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 4)
    
    imageContours, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #cv2.drawContours(img, contours, -1, (0,255,0), 1, cv2.LINE_AA)      

    
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
            #cv2.rectangle(img, (x-2,y-2), (x+w,y+h), (255, 0, 0), 2)
            solidez = float(area)/hull_area    
            
           
            
            chars = img[y:y+h,x:x+w]               
            caracteres.append(chars)
            
            
            '''numero += 1                                   
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
                  "-------------------------------------------------------",'\n')'''
                    
    
    print ('caracteres: ',len(caracteres), '\n')
    return caracteres


for i in range(1,2):     
    
    path = 'D:\Documents\OPENCV\Placas\Placa ('+str(i)+").jpg"
    img = cv2.imread(path)
    print("imagen ", i)
    caracteres = characters(path)
    numero = 0
    another = 0
    
    
        
    cv2.namedWindow('PLACA', cv2.WINDOW_NORMAL)
    cv2.imshow('PLACA', img)
    cv2.waitKey()
    
    
    
    for x in caracteres:
        numero += 1
        another += 7
        
        '''cv2.namedWindow(str(numero), cv2.WINDOW_NORMAL)
        cv2.imshow(str(numero), x)
        cv2.waitKey()'''
        
        sourcer = FeatureSourcer(sourcer_params, x)

        f = sourcer.features(x)
        print("feature shape:", f.shape)

        rgb_img, a_img, b_img, c_img = sourcer.visualize()
        show_images([rgb_img, a_img, b_img, c_img], per_row = 4, per_col = 1, W = 10, H = 2)
        
        
        '''hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                                histogramNormType,L2HysThreshold,gammaCorrection,nlevels, useSignedGradients)
        
        descriptor = hog.compute(x)
        
      
        
        print ('features: ',len(descriptor), '\n')'''
        
        ABC_img = convert(x, src_model= 'rgb', dest_model = 'ycrcb')
        
        cv2.namedWindow(str(numero), cv2.WINDOW_NORMAL)
        cv2.imshow(str(numero), x)
        cv2.waitKey()
        
        fig, ax = plt.subplots(1,2)
        fig.set_size_inches(8,6)
        # remove ticks and their labels
        [a.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
            for a in ax]

        ax[0].imshow(x)
        ax[0].set_title('dog')
        ax[1].imshow(c_img)
        ax[1].set_title('hog')
        plt.show()
        
        
        
        

        
        
    
    cv2.destroyAllWindows()