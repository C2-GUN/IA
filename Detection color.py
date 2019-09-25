import numpy as np
import cv2
from matplotlib import pyplot as plt
import argparse

image = cv2.imread("D:\Documents\OPENCV\DB_PLATES\carro (10).jpg", 1)

crop = image[y:y+h, x:x+w]
cv2.imshow("cropped", crop)
cv2.waitKey(0)

#image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = cv2.GaussianBlur(gray, (5, 5), 0)

t, thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)

img2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(image, contours, -1, (0,255,0), 4, cv2.LINE_AA)

for cnt in contours:
    approx = cv2.approxPolyDP(cnt,0.02*cv2.arcLength(cnt,True),True)
    if len(approx)==4:
        x,y,w,h = cv2.boundingRect(cnt)
        diag = h/w
        #if(diag >= 0.3 and diag <= 0.48):        
        area = cv2.contourArea(cnt)
        ar = float(w)/float(h)
        cv2.drawContours(image,[cnt],0,(0,0,255),-1)
        cv2.rectangle(image, (x,y), (x+w,y+h), (255, 0, 255), 7)
        
            
        print("X: ", x , "Y: ", y, "W: ", w , "H: ", h, y+h, x+w, h/w)
        
    elif len(approx)==3:
        print ("triangle")
        cv2.drawContours(image,[cnt],0,(0,255,0),-1)
    elif len(approx)==4:
        print ("square")
        cv2.drawContours(image,[cnt],0,(0,0,255),-1)
    elif len(approx) == 9:
        print ("half-circle")
        cv2.drawContours(image,[cnt],0,(255,255,0),-1)
    elif len(approx) > 15:
        print ("circle")
        cv2.drawContours(image,[cnt],0,(0,255,255),-1)
    elif len(approx)==5:
        print ("pentagon")
        cv2.drawContours(image,[cnt],0,255,-1)
        




        
                


cv2.namedWindow('Grises', cv2.WINDOW_NORMAL)
cv2.imshow('Grises',gray)
cv2.waitKey()

cv2.namedWindow('Treshold', cv2.WINDOW_NORMAL)
cv2.imshow('Treshold',thresh)
cv2.waitKey()

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image',image)
#cv2.imwrite('holi.jpg',Placa)
cv2.waitKey()

cv2.destroyAllWindows()

#cv2.imshow("Ventana de imagen", dst)
#cv2.waitKey(0)

