import cv2
from matplotlib import pyplot as plt

path ='D:\Documents\OPENCV\DB_PLATES\carro (69).jpg'

print('PATHHHHHHHHH: ', path[32:])

image = cv2.imread(path)

folder = 'D:\\Documents\\OPENCV\\'
name = "Placa_" + path[32:]
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = cv2.GaussianBlur(gray, (5, 5), 0)

t, thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)

img2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(image, contours, -1,(0,255,0), 4, cv2.LINE_AA)

for cnt in contours:
    approx = cv2.approxPolyDP(cnt,0.02*cv2.arcLength(cnt,True),True)
    if len(approx)==4:
        x,y,w,h = cv2.boundingRect(cnt)        
        area = cv2.contourArea(cnt)
        ar = float(w)/float(h)
        if(y+h>1000 and y+h<=2100):       
            if x+w>1000 and x+w <= 2000:
                if(area >= 50000):
                    cv2.drawContours(image,[cnt],0, (0,0,255),-1)
                    cv2.rectangle(image, (x,y), (x+w,y+h), 0)
                    print("X: ", x , "Y: ", y, "W: ", w , "H: ", h, y+h, x+w, )
                    Placa = image[y:y+h,x:x+w]
                    cv2.imwrite('%s/%s.jpg' % (folder,name), Placa)

   
                


cv2.namedWindow('Grises', cv2.WINDOW_NORMAL)
cv2.imshow('Grises',gray)
cv2.waitKey()

cv2.namedWindow('Treshold', cv2.WINDOW_NORMAL)
cv2.imshow('Treshold',thresh)
cv2.waitKey()

cv2.namedWindow('Treshold', cv2.WINDOW_NORMAL)
cv2.imshow('Treshold',image)
cv2.waitKey()

cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
cv2.imshow('Output',Placa)
cv2.waitKey()
#cv2.imwrite('holi.jpg',Placa)

cv2.destroyAllWindows()

#cv2.imshow("Ventana de imagen", dst)
#cv2.waitKey(0)

