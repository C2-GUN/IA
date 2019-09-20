# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 19:59:46 2019

@author: CDEC
"""

import glob


chars = glob.glob('D:\Documents\OPENCV\TRAINING')

from pathlib import Path

'''for filename in glob.glob('*'):
    print(filename)
    
print('-----------------------')    

for filename in Path('D:\Documents\OPENCV\TRAINING').glob('*'):
    print(filename)


print('-----------------------') '''


for caracter in glob.glob('D:\Documents\OPENCV\TRAINING' +'\*'):
    print('\n')
    print(caracter)
 
    for x in Path(caracter).glob('*.jpg'):
        print(x)

print('a')
       