# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 00:07:18 2019

@author: CDEC
"""

import cv2
import numpy as np
kernel = np.ones((5,5),np.uint8)
import matplotlib.pyplot as plt
import glob
from pathlib import Path
from sklearn.model_selection import  train_test_split
from sklearn.externals import joblib
from sklearn import metrics
from skimage.feature import haar_like_feature
from skimage.transform import integral_image
from timeit import default_timer as timer
from sklearn.metrics import confusion_matrix
import scikitplot as skplt
from sklearn.ensemble import RandomForestClassifier

start = timer()

folder = 'D:\Documents\OPENCV\TRAINING'
width = 40
height = 40
dim = (width, height)
label_list = []
features_list = []
features_windows = []
window_feature_IN_piramid = []
window = []
feature_types = ['type-4']
windows = []


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
      img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else: 
      raise Exception('ERROR:', 'No se puede cambiar de color')
    
    '''hogA_img = img[:, :, 0]
    hogB_img = img[:, :, 1]
    hogC_img = img[:, :, 2]'''
    
    return img#, hogA_img, hogB_img, hogC_img


def sliding_window_(image, stepSize, windowSize):	
    for y in range(0, image.shape[0]):
        for x in range(0, image.shape[1]):			
            window = image[x:x + windowSize, y:y + windowSize]
            cv2.namedWindow('Digito1', cv2.WINDOW_NORMAL)
            cv2.imshow('Digito1', window)
            clone = image.copy()
            cv2.rectangle(image, (x, y), (x + windowSize, y + windowSize), (0, 255, 0), 0)
            cv2.namedWindow('Digito2', cv2.WINDOW_NORMAL)
            cv2.imshow('Digito2', clone)
            cv2.waitKey()
            cv2.destroyAllWindows()
            try_window_feature = integral_image(window)
            features = haar_like_feature(try_window_feature, 0, 0, 5, 5, feature_types)
            print(features)
            print('sliding window')
            print(len(features))
            features_windows.append(features)
            print(len(features_windows))
            '''windows.append(window)
            print('cantidad de ventanas: ', len(windows))'''            
            
            
            
            
    return features_windows     

def pyramid(image, minSize=(20, 20)):
    flag = True
    while flag == True:

        print('IN')
        
        '''cv2.namedWindow('Digito0', cv2.WINDOW_NORMAL)
        cv2.imshow('Digito0', image)
        print(image.shape)
        cv2.waitKey()
        cv2.destroyAllWindows()'''
        window_feature_IN_piramid = sliding_window_(image, 10, 10)
        print('pyramid')
        print(len(window_feature_IN_piramid))
        
        
        if image.shape[0] <= minSize[1] or image.shape[1] <= minSize[0]:
            flag = False
        else: 
            W = image.shape[0] - 10
            H = image.shape[1] - 10
            dim = (W,H)
            image = cv2.resize(image, dim)
		
        print(flag)
    return window_feature_IN_piramid

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    

    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax  



 





for carpetas in glob.glob('D:\Documents\OPENCV\TRAINING' +'\*'):
    print('\n')
    print(carpetas)
    
    print(label_list)
    
    
 
    for digitos in Path(carpetas).glob('*.jpg'):
        img = cv2.imread(str(digitos))
        '''cv2.namedWindow('PLACA', cv2.WINDOW_NORMAL)
        cv2.imshow('PLACA', img)
        cv2.waitKey()'''
        
        
        imgGREY = change_color(img, sourcer_params)
        imgX = integral_image(imgGREY)
        
                
        features = haar_like_feature(imgX, 0, 0, 40, 40, feature_types)
        #feature_coord, feat_type = haar_like_feature_coord(20, 20, 'type-4')
        #imagedraw = draw_haar_like_feature(imgX, 0, 0, 40, 40, feature_coord)
        
        '''cv2.namedWindow('Digito grey', cv2.WINDOW_NORMAL)
        cv2.imshow('Digito grey', imgX)
        cv2.waitKey()
        
        cv2.namedWindow('HOG image', cv2.WINDOW_NORMAL)
        cv2.imshow('HOG image', hog_img)
        cv2.waitKey()'''
        
        print('Caracteristicas: ' ,len(features))
        
        fig, ax = plt.subplots(1,2)
        fig.set_size_inches(8,6)
        # remove ticks and their labels
        [a.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
            for a in ax]

        ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax[0].set_title('caracter')
        ax[1].imshow(imgGREY,'gray')
        ax[1].set_title('Escala grises')
        

        

        plt.show()
       
        
        features_list.append(features)
        label_list.append(str(carpetas[29:30]))
        print('Caracteristicas totales: ' ,len(features_list) )
        print('Labels totales: ' ,len(label_list) )
        
        #input("Press Enter to continue...")
        

features = np.array(features_list)
X_train, X_test, y_train, y_test= train_test_split(features, label_list, test_size=0.3)

clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

model_score = clf.score(X_test, y_test)
y_pred = clf.predict(X_test)
joblib.dump(clf, 'D:\Documents\OPENCV\MODELS\HAAR_RANDOMFOREST_MODEL_'+str(metrics.accuracy_score(y_test, y_pred))+'.pkl')
end = timer()
print("{0:.3f}".format(end - start)+' seconds') # Time in seconds
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
plt.figure(figsize=(20,20))
plot_confusion_matrix(y_test,y_pred,['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'])
plt.savefig('CONFUSION_MATRIX_HAAR_RANDOMFOREST')

'''skplt.metrics.plot_roc_curve(y_test, y_pred)
plt.show()'''