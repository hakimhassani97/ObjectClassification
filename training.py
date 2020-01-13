import cv2
import numpy as np
from skimage import exposure
from skimage import feature
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imutils import paths
import argparse
import cv2
import os
import imutils

def extract_color_histogram(image, bins=(8, 8, 8)):
	# extract a 3D color histogram from the HSV color space using
	# the supplied number of `bins` per channel
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,[0, 180, 0, 256, 0, 256])
	# handle normalizing the histogram if we are using OpenCV 2.4.X
	if imutils.is_cv2():
		hist = cv2.normalize(hist)
	# otherwise, perform "in place" normalization in OpenCV 3 (I
	# personally hate the way this is done
	else:
		cv2.normalize(hist, hist)
	# return the flattened histogram as the feature vector
	return hist.flatten()

# RGB data folder
dataFolder='data/dataRGB/'
# Binary data folder
# dataFolder='data/dataBinary/'
# size of data images
imgSize=(76,76)
# classes
classes=['bike','boats','canoe','car','human','noise','pickup','truck','van']

# grab the list of images that we'll be describing
print("[INFO] describing images...")
# initialize the data matrix and labels list
data = []
labels = []

# loop over the input images
for c in classes:
    files=os.listdir(dataFolder+c)
    for file in files:
        img = cv2.imread(dataFolder+c+'/'+file)
        img=cv2.resize(img,imgSize,interpolation = cv2.INTER_AREA)
        (H, hogImage) = feature.hog(img, orientations=9, pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1",
            visualize=True)
        data.append(np.asarray(H))
        labels.append(c)
        # hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
        # hogImage = hogImage.astype("uint8")
        # cv2.imshow("HOG Image", hogImage)
        # cv2.waitKey()

# encode the labels, converting them from strings to integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# partition the data into training and testing splits, using 75%
# of the data for training and the remaining 25% for testing
print("[INFO] constructing training/testing split...")
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(data), labels, test_size=0.25, random_state=42)
 
# train the linear regression clasifier
print("[INFO] training Linear SVM classifier...")
model = LinearSVC()
model.fit(trainData, trainLabels)
# now you can save it to a file
import pickle
with open('svmClassifier.pkl', 'wb') as f:
    pickle.dump((le,model), f)

# evaluate the classifier
print("[INFO] evaluating classifier...")
predictions = model.predict(testData)
print(classification_report(testLabels, predictions,target_names=le.classes_))










#######################################################################
# hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
# hogImage = hogImage.astype("uint8")
# print()
# print(np.shape(hogImage))

# cv2.imshow("HOG Image", hogImage)
# cv2.waitKey()

# img = cv2.imread('data/dataRGB/bike/Pedestrain_4_23.png')
# img = cv2.imread('lenna.jpg')
# gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# sift = cv2.xfeatures2d.SIFT_create()
# kp = sift.detect(gray,None)

# img=cv2.drawKeypoints(gray,kp,img)
# cv2.imwrite('sift_keypoints.jpg',img)