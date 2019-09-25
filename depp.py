# -*- coding: utf-8 -*-
"""
Created on Thu May 30 21:43:05 2019

@author: CDEC
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.svm import SVC
x = [2, 6, 1.5, 9, 3, 13]
y = [2, 9, 1.8, 8, 0.6, 11]
plt.scatter(x,y)
plt.show()
X = np.array([[2,2],
[6,9],
[1.5,1.8],
[9,8],
[3,0.6],
[13,11]])
y = [0,1,0,1,0,1]
clf = SVC(kernel='linear', probability=True, tol=1e-3)
clf.fit(X,y)
j=clf.predict([[0,1]])
print(j)

