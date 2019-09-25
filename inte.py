import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure
import cv2


txt = "apple#banana#cherry#orange"

x = txt.split("#")[-2]

print(x)