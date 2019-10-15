# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 22:31:08 2019

@author: CDEC
"""

import cv2
import numpy as np
from skimage.feature import hog
import matplotlib.pyplot as plt
import glob
from pathlib import Path
from sklearn.model_selection import  train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from sklearn import metrics
from sklearn import svm
from sklearn.metrics import confusion_matrix
from timeit import default_timer as timer
import scikitplot as skplt

start = timer()


folder = 'D:\Documents\OPENCV\TRAINING'
width = 40
height = 40
dim = (width, height)
label_list = []
features_list = []

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


def Hoog(img, sourcer_params):
         
    features, hog_img = hog(img, 
                        orientations = sourcer_params['number_of_orientations'], 
                        pixels_per_cell = (sourcer_params['pixels_per_cell'], sourcer_params['pixels_per_cell']),
                        cells_per_block = (sourcer_params['cells_per_block'], sourcer_params['cells_per_block']), 
                        transform_sqrt = sourcer_params['transform_sqrt'], 
                        visualise = True, 
                        feature_vector = True,
                        block_norm='L2-Hys')
    

    return features, hog_img


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
        print(digitos) 
        img = cv2.imread(str(digitos))
        '''cv2.namedWindow('PLACA', cv2.WINDOW_NORMAL)
        cv2.imshow('PLACA', img)
        cv2.waitKey()'''
        
        
        imgX = change_color(img, sourcer_params)
        
        (features, hog_img) = Hoog(imgX, sourcer_params)
        
        '''cv2.namedWindow('Digito grey', cv2.WINDOW_NORMAL)
        cv2.imshow('Digito grey', imgX)
        cv2.waitKey()
        
        cv2.namedWindow('HOG image', cv2.WINDOW_NORMAL)
        cv2.imshow('HOG image', hog_img)
        cv2.waitKey()'''
        
        print('Caracteristicas: ' ,len(features) )
        
        fig, ax = plt.subplots(1,3)
        fig.set_size_inches(8,6)
        # remove ticks and their labels
        [a.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
            for a in ax]

        ax[0].imshow(img)
        ax[0].set_title('caracter')
        ax[1].imshow(imgX)
        ax[1].set_title('Escala grises')
        ax[2].imshow(hog_img)
        ax[2].set_title('HOG')
        

        plt.show()
       
        
        features_list.append(features)
        label_list.append(str(carpetas[29:30]))
        print('Caracteristicas totales: ' ,len(features_list) )
        
        #input("Press Enter to continue...")
        

features = np.array(features_list, 'float64')
X_train, X_test, y_train, y_test = train_test_split(features, label_list, test_size=0.3, random_state=109)

clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma=0.01, kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
clf.fit(X_train, y_train)

model_score = clf.score(X_test, y_test)

y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred, average='micro'))
print("Recall:",metrics.recall_score(y_test, y_pred, average='micro'))

joblib.dump(clf, 'D:\Documents\OPENCV\MODELS\HOG_SVM_MODEL_'+str(metrics.accuracy_score(y_test, y_pred))+'.pkl')
end = timer()
print("{0:.3f}".format(end - start)+' seconds') # Time in seconds
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
plt.figure(figsize=(20,20))
plot_confusion_matrix(y_test,y_pred,['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'])
plt.savefig('CONFUSION2_MATRIX_HOG_SVM')
skplt.metrics.plot_roc_curve(y_test, y_pred)
plt.show()