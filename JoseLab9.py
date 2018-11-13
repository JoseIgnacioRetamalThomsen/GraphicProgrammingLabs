import cv2
import numpy as np 
from matplotlib import pyplot as plt

img = cv2.imread('studentPicture.jpg',) 




gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



nrows=3
ncols =3

#add normal image
plt.subplot(nrows, ncols,1),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Original')
plt.xticks([])
plt.yticks([]) 
#add grayscale image
plt.subplot(nrows, ncols,2),plt.imshow(gray, cmap = 'gray')
plt.title("GrayScale")
plt.xticks([])
plt.yticks([]) 


#blur image 1
KernelSizeWidth = 5
KernelSizeHeight = 5
blur1 = cv2.GaussianBlur(gray,(KernelSizeWidth, KernelSizeHeight),0) 

#ad blur1 image
plt.subplot(nrows, ncols,3),plt.imshow(blur1, cmap = 'gray')
plt.title("Blur image at 5")
plt.xticks([])
plt.yticks([])

#blur 2
KernelSizeWidth = 13
KernelSizeHeight = 13
blur2 = cv2.GaussianBlur(gray,(KernelSizeWidth, KernelSizeHeight),0) 

#add blur 2
plt.subplot(nrows, ncols,4),plt.imshow(blur2, cmap = 'gray')
plt.title("Blur image at 13")
plt.xticks([])
plt.yticks([])

sobelHorizontal = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)  # x dir 
sobelVertical   = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)  # y dir

#sobel horizontal
plt.subplot(nrows, ncols,5),plt.imshow(sobelHorizontal, cmap = 'gray')
plt.title("sobel Horizontal")
plt.xticks([])
plt.yticks([])

#sobel vertical
plt.subplot(nrows, ncols,6),plt.imshow(sobelVertical, cmap = 'gray')
plt.title("sobel vertical")
plt.xticks([])
plt.yticks([])

#add 2 sobel
doubleSobel = sobelHorizontal + sobelVertical

#double
plt.subplot(nrows, ncols,7),plt.imshow(doubleSobel, cmap = 'gray')
plt.title("2 sobel")
plt.xticks([])
plt.yticks([])

#canny
cannyThreshold = 50
cannyParam2  = 150
canny = cv2.Canny(gray,cannyThreshold,cannyParam2)
 
#show
plt.subplot(nrows, ncols,8),plt.imshow(canny, cmap = 'gray')
plt.title("canny 100-200")
plt.xticks([])
plt.yticks([])

#treshold
tresshold = 120

b = doubleSobel
r=0
c=0

height, width = doubleSobel.shape

for x in range(0,height):
    for y in range(0,width):
        if (doubleSobel[x][y] > tresshold):
            doubleSobel[x][y]=255
        else:
            doubleSobel[x][y]=0

              
        

#show
plt.subplot(nrows, ncols,9),plt.imshow(doubleSobel, cmap = 'gray')
plt.title("canny 100-200")
plt.xticks([])
plt.yticks([])

plt.show()
cv2.waitKey(0)