# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 05:37:24 2019

@author: CDEC
"""

import numpy as np

def funcion1(x):
    print('hola')
    print(x)
    print(x.shape[0])
    if(x.shape[0] == x.shape[1]):
        print('la matriz es simetrica')
    else:
        print('matrix antisimetrica')

X = np.array([[11 ,12,13,11], [21, 22, 23, 24], [31,32,33,34]])
Y = np.array([[11 ,12,13,11], [21, 22, 23, 24], [31,32,33,34], [21, 22, 23, 24]])



funcion1(X)
funcion1(Y)