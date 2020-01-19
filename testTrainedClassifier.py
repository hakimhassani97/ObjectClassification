import sklearn.svm
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from skimage import exposure
from skimage import feature
import pickle
import cv2
import os

# and later you can load it
with open('savedModels/svmClassifierRGB.pkl', 'rb') as f:
    le,model = pickle.load(f)

# RGB data folder
dataFolder='data/dataRGB/'
# Binary data folder
# dataFolder='data/dataBinary/'
# size of data images
imgSize=(76,76)
files=os.listdir('newTestImages/')
for file in files:
    # load the image to test
    img = cv2.imread('newTestImages/'+file)
    img=cv2.resize(img,imgSize,interpolation = cv2.INTER_AREA)
    (H, hogImage) = feature.hog(img, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1",
        visualize=True)
    # hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
    # hogImage = hogImage.astype("uint8")

    # predict
    labels=['bike','boats','canoe','car','human','noise','pickup','truck','van']
    # labels=le.transform(labels)
    prediction=model.predict(np.asarray(H).reshape(1, -1))
    label=labels[prediction[0]]
    # print(label)
    # print label Text
    img = cv2.imread('newTestImages/'+file)
    newSize=(np.shape(img)[1]*2,np.shape(img)[0]*2)
    img=cv2.resize(img,newSize,interpolation = cv2.INTER_AREA)
    cv2.rectangle(img,(0,0),(70,15),(0,255,0),-1)
    cv2.putText(img,label,(10,10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
    cv2.imshow("Press any key for next image", img)
    cv2.waitKey()
    # print(classification_report('bike', prediction,target_names=le.classes_))