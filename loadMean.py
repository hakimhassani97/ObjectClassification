import cv2
import numpy as np
from skimage import exposure
from skimage import feature
from sklearn.svm import LinearSVC
import cv2
import os
import imutils

# RGB data folder
dataFolder='data/dataRGB/bike/'
# Binary data folder
dataFolder='data/dataBinary/bike/'
# size of data images
imgSize=(76,76)

classes=['bike','boats','canoe','car','human','noise','pickup','truck','van']

classesMed=[]
img = cv2.imread(dataFolder+'Pedestrain_4_23.png',cv2.IMREAD_GRAYSCALE)
img=cv2.resize(img,imgSize,interpolation = cv2.INTER_AREA)
for c in classes:
    med=np.loadtxt('means/'+c+'.txt')
    med=np.subtract(med,img)
    print(c,np.sum(med))

print(np.shape(img))
print(np.shape(med))

cv2.imshow("HOG Image", med)
cv2.waitKey()