# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 05:48:25 2019

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
from timeit import default_timer as timer
from skimage.feature import local_binary_pattern
from skimage import feature
from skimage.feature import multiblock_lbp
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.svm import LinearSVC

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


    


def descriptor(image, numPoints, radius, eps=1e-7):
    # compute the Local Binary Pattern representation
	# of the image, and then use the LBP representation
	# to build the histogram of patterns
	lbp = feature.local_binary_pattern(image, numPoints, radius, method="uniform")
	(hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, numPoints + 3),
			range=(0, numPoints + 2))
 
	# normalize the histogram
	hist = hist.astype("float")
	hist /= (hist.sum() + eps)
 
	# return the histogram of Local Binary Patterns
	return lbp, hist


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
        
        features = local_binary_pattern(imgGREY, 14, 2, method="default")
        lbp, hist = descriptor(imgGREY, numPoints = 24 , radius = 8, eps=1e-7)
        

        
           
        
        '''cv2.namedWindow('Digito grey', cv2.WINDOW_NORMAL)
        cv2.imshow('Digito grey', lbp)
        cv2.waitKey()
        
        cv2.namedWindow('HOG image', cv2.WINDOW_NORMAL)
        cv2.imshow('HOG image', lbp)
        cv2.waitKey()'''
        
        print('Caracteristicas: ' ,len(hist))
        
        fig, ax = plt.subplots(1,5)
        fig.set_size_inches(8,6)
        # remove ticks and their labels
        [a.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
            for a in ax]

        ax[0].imshow(img)
        ax[0].set_title('caracter')
        ax[1].imshow(imgGREY)
        ax[1].set_title('Escala grises')
        ax[2].imshow(features)
        ax[2].set_title('LBP')
        ax[3].set_xlim([0, 256])
        ax[3].set_ylim([0, 0.030]) 
        ax[3].hist(features.ravel(), normed=True, bins=20, range=(0, 256))               
        ax[3].set_title('Histogram')
        ax[4].imshow(lbp)
        ax[4].set_title('LBP2')
        

        plt.show()
       
        
        features_list.append(hist)
        label_list.append(str(carpetas[29:30]))
        print('Caracteristicas totales: ' ,len(features_list) )
        print('Labels totales: ' ,len(label_list) )
        
        #input("Press Enter to continue...")
        

features = np.array(features_list, 'float64')
X_train, X_test, y_train, y_test= train_test_split(features, label_list, test_size=0.3)




'''clf = svm.SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma=0.01, kernel='linear',
    max_iter=-1, probability=False, random_state=42, shrinking=True,
    tol=0.001, verbose=False)
clf.fit(X_train, y_train)'''

clf = LinearSVC(C=100.0, random_state=42)
clf.fit(X_train, y_train)

model_score = clf.score(X_test, y_test)

y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred, average='micro'))
print("Recall:",metrics.recall_score(y_test, y_pred, average='micro'))
print("model score:",model_score)

joblib.dump(clf, 'D:\Documents\OPENCV\MODELS\LBP_SVM_MODEL_'+str(metrics.accuracy_score(y_test, y_pred))+'.pkl')
end = timer()
print("{0:.3f}".format(end - start)+' seconds') # Time in seconds


plt.figure(figsize=(20,20))
plot_confusion_matrix(y_test,y_pred,['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'])
plt.savefig('CONFUSION_MATRIX_LBP_SVM_fue')
