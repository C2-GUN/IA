# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 06:39:56 2019

@author: CDEC
"""

import cv2
import numpy as np
kernel = np.ones((5,5),np.uint8)
from skimage import exposure
from skimage import feature
from skimage.feature import hog
import matplotlib.pyplot as plt

folder = 'D:\Documents\OPENCV\TRAINING'
width = 40
height = 40
dim = (width, height)

sourcer_params = {
  'color_model': 'grey',                # hls, hsv, yuv, ycrcb
  'bounding_box_size': 64,             #
  'number_of_orientations': 12,        # 6 - 12
  'pixels_per_cell': 8,               # 8, 16
  'cells_per_block': 2,                # 1, 2
  'transform_sqrt': True
}

def change_color(img, sourcer_params):
    
    
    if sourcer_params['color_model'] == "hsv": 
      img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif sourcer_params['color_model'] == "hls":
      img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    elif sourcer_params['color_model'] == "yuv":
      img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    elif sourcer_params['color_model'] == "ycrcb":
      img = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
    elif sourcer_params['color_model'] == "grey":
      img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    else: 
      raise Exception('ERROR:', 'No se puede cambiar de color')
    
    hogA_img = img[:, :, 0]
    hogB_img = img[:, :, 1]
    hogC_img = img[:, :, 2]
    
    return img, hogA_img, hogB_img, hogC_img


def Hoog(img, sourcer_params):
         
    features, hog_img = hog(img, 
                        orientations = sourcer_params['number_of_orientations'], 
                        pixels_per_cell = (sourcer_params['pixels_per_cell'], sourcer_params['pixels_per_cell']),
                        cells_per_block = (sourcer_params['cells_per_block'], sourcer_params['cells_per_block']), 
                        transform_sqrt = sourcer_params['transform_sqrt'], 
                        visualise = True, 
                        feature_vector = True)
    

    return features, hog_img


 

def characters(path):
    img = cv2.imread(path)
    
    numero = 0    
    
    caracteres = []
    
       
    
    
   
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 4)
    
    imageContours, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP & cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #  cv2.RETR_EXTERNAL & cv2.RETR_CCOMP 
    # cv2.RETR_CCOMP & cv2.RETR_EXTERNAL
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
        
        
        
        
        if(aspect_ratio > 0.1 and aspect_ratio < 1 and area >= 650 and equi_diametro >= 32 and extension >= 0.1):
            (x, y, w, h) = cv2.boundingRect(cnt)
            #cv2.drawContours(img,[cnt],0,(0,0,255),-1,4)
            #cv2.rectangle(img, (x-2,y-2), (x+w,y+h), (255, 0, 0), 2)
                   
            
            numero += 1  
            
            
            
            chars = img[y:y+h,x:x+w]
            
            # resize image
            chars = cv2.resize(chars, dim, interpolation = cv2.INTER_AREA)
            
            #name = str(numero) + '.jpg'
            #cv2.imwrite('%s/%s' % (folder,name), chars)               
            caracteres.append(chars)
            '''print(hierarchy)
            print('----------------')'''
            
            
                        
            
            
            
            
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


for i in range(193,194):     
    
    path = 'D:\Documents\OPENCV\Placas\Placa ('+str(i)+").jpg"
    img = cv2.imread(path)
    print("imagen ", i)
    caracteres = characters(path)
    numero = 0
    another = 0
    another1 = 0
    Total_features = []
    
    (imgX, A_img, B_img, C_img) = change_color(img, sourcer_params)
        
    (A_features, hog_A_img) = Hoog(A_img, sourcer_params)
 
        
    cv2.namedWindow('PLACA', cv2.WINDOW_NORMAL)
    cv2.imshow('PLACA', img)
    cv2.waitKey()
    
    

    
    '''(H, hogImage) = feature.hog(img, orientations=12, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", visualise=True)
    hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
    hogImage = hogImage.astype("uint8")'''
 
    #cv2.imshow("HOG Image", hogImage)
    
    
    
    for chars in caracteres:
    
        
        
        numero += 1
        another += 7
        another1 += 8
        
        '''x = cv2.cvtColor(chars, cv2.COLOR_BGR2GRAY)
        
        fd, hog_image = hog(x, orientations=12, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualise=True,  transform_sqrt=True)'''
        
        
        
        x = cv2.cvtColor(chars, cv2.COLOR_BGR2GRAY)
        (imgX, A_img, B_img, C_img) = change_color(chars, sourcer_params)
        
        (A_features, hog_A_img) = Hoog(A_img, sourcer_params)
        (B_features, hog_B_img) = Hoog(B_img, sourcer_params)
        (C_features, hog_C_img) = Hoog(C_img, sourcer_params)
        (imgX_features, hog_imgx) = Hoog(x, sourcer_params)
        
        Total_features = np.hstack((A_features, B_features, C_features))
        
        
        '''cv2.namedWindow(str(numero), cv2.WINDOW_NORMAL)
        cv2.imshow(str(numero), imgX)
        cv2.waitKey()
        
        cv2.namedWindow(str(another), cv2.WINDOW_NORMAL)
        cv2.imshow(str(another), A_img)
        cv2.waitKey()
        
        cv2.namedWindow(str(another1), cv2.WINDOW_NORMAL)
        cv2.imshow(str(another1), B_img)
        cv2.waitKey()'''
        
        print(len(A_features))
        print(len(B_features))
        print(len(C_features))
        print(len(Total_features))
        print(len(imgX_features))
        
        fig, ax = plt.subplots(1,4)
        fig.set_size_inches(8,6)
        # remove ticks and their labels
        [a.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
            for a in ax]

        ax[0].imshow(chars)
        ax[0].set_title('caracter')
        ax[1].imshow(hog_A_img)
        ax[1].set_title('hog_A_img')
        ax[2].imshow(hog_B_img)
        ax[2].set_title('hog_B_img')
        ax[3].imshow(hog_C_img)
        ax[3].set_title('hog_C_img')

        plt.show()
        
        
        
        
        '''hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                                histogramNormType,L2HysThreshold,gammaCorrection,nlevels, useSignedGradients)
        
        descriptor = hog.compute(x)'''
        
        
        
        '''(H, hogImage) = feature.hog(x, orientations=9, pixels_per_cell=(8, 8),
                 cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", visualise=True)
        hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
        hogImage = hogImage.astype("uint8")'''
 
        #cv2.imshow("HOG Image", hogImage)
        
        
        
        
        
        
      
        
        
        
        
        
        

        
        
    
    cv2.destroyAllWindows()