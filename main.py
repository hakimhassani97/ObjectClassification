import cv2
import numpy as np
import time
from scipy import ndimage

# functions
def difference(imgC,imgP):
    res=np.copy(imgC)
    # win_mean = ndimage.uniform_filter(res, (h, w))
    # win_sqr_mean = ndimage.uniform_filter(res**2, (h, w))
    # var = win_sqr_mean - win_mean**2
    for i in range(0,h):
        for j in range(0,w):
            d=abs(float(imgC[i,j])-float(imgP[i,j]))
            # print(imgP[i,j])
            # res[i,j]=d
            if d>threshold:#*var[i,j]:
                res[i,j]=255
            else:
                res[i,j]=0
    return res

def mediane(frames):
    # if len(frames)>100:
    #     frames.remove(frames[0])
    #     print(len(frames))
    # Convert images to 4d ndarray, size(n, nrows, ncols, 3)
    frames = np.asarray(frames)
    # Take the median over the first dim
    med = np.mean(frames, axis=0)
    return med

# init
p='boats/1.png'
c='boats/2.png'
threshold=50
blurSize=16
medianSize=15
nbFrames=1031

imgP=cv2.imread('data/'+p,cv2.IMREAD_GRAYSCALE)
imgC=cv2.imread('data/'+c,cv2.IMREAD_GRAYSCALE)
frames=[imgP]
median=np.loadtxt('gaussians/boatsGauss.txt')
h,w=np.shape(imgP)

res=np.copy(imgC)
start_time=time.time()

# apply gaussian blur
# imgP = cv2.blur(imgP,(blurSize,blurSize))
# imgC = cv2.blur(imgC,(blurSize,blurSize))
# apply medianBlur
# imgP = cv2.medianBlur(imgP,medianSize)
# imgC = cv2.medianBlur(imgC,medianSize)

for i in range(2,nbFrames+1):
    imgC=cv2.imread('data/boats/'+str(i)+'.png',cv2.IMREAD_GRAYSCALE)
    # median=mediane(frames)
    res=difference(imgC,median)
    cv2.imshow('result',res)
    if cv2.waitKey(33) == ord('q'):
        print("exited")
        break
    # if i==nbFrames-1:
    #     np.savetxt('gaussians/median.txt',median)
    #     print('file saved as median.txt')
    # frames.append(imgC)

print("------- %s seconds -------" % (time.time() - start_time))
cv2.destroyAllWindows()
