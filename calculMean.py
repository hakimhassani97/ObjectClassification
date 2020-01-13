import cv2
import numpy as np
from skimage import exposure
from skimage import feature
from sklearn.svm import LinearSVC
import cv2
import os
import imutils

classes=['bike','boats','canoe','car','human','noise','pickup','truck','van']
className='bike'
# RGB data folder
dataFolder='data/dataRGB/'+className+'/'
# Binary data folder
dataFolder='data/dataBinary/'+className+'/'
# size of data images
imgSize=(76,76)

orientations=[]
files=os.listdir(dataFolder)
for file in files:
    img = cv2.imread(dataFolder+file)
    img=cv2.resize(img,imgSize,interpolation = cv2.INTER_AREA)
    (H, hogImage) = feature.hog(img, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1",
        visualize=True)
    orientations.append(hogImage)
    hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
    hogImage = hogImage.astype("uint8")
    # cv2.imshow("HOG Image", img)
    # cv2.imshow("HOG Image", hogImage)
    # cv2.waitKey()
# Convert images to 4d ndarray, size(n, nrows, ncols, 3)
orientations = np.asarray(orientations)
# Take the median over the first dim
med = np.mean(orientations, axis=0)
np.savetxt(className+'.txt',med)

cv2.imshow("HOG Image", med)
cv2.waitKey()