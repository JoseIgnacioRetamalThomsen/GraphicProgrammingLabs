import cv2
import numpy as np 
from matplotlib import pyplot as plt
from drawMatches import drawMatches


#open image
img = cv2.imread('GMIT1.jpg',)

#create gray scale image from img
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#plot boundaries
nrows=3
ncols =4

#add normal image
plt.subplot(nrows, ncols,1),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Normal')
plt.xticks([])
plt.yticks([]) 
#add grayscale image
plt.subplot(nrows, ncols,2),plt.imshow(grayImg, cmap = 'gray')
plt.title("GrayScale")
plt.xticks([])
plt.yticks([]) 


#Harris corner detection fro gray scale image

##create deep compy
harrisImg = grayImg.copy()

blockSize= 2
aperture_size = 3
k = 0.04

#perform harris
dst = cv2.cornerHarris(harrisImg, blockSize, aperture_size, k)

#add haris to display
plt.subplot(nrows, ncols,3),plt.imshow(dst, cmap = 'gray')
plt.title("haris not circles")
plt.xticks([])
plt.yticks([]) 

##draw circles
B=100
G =10
R =200

#copy of color image to draw circles
circlesHarry = img.copy()

threshold = 0.09; #number between 0 and 1 
for i in range(len(dst)):   
    for j in range(len(dst[i])):   
        if dst[i][j] > (threshold*dst.max()):    
            cv2.circle(circlesHarry,(j,i),3,(B, G, R),-1)

#add haris to display
plt.subplot(nrows, ncols,4),plt.imshow(cv2.cvtColor(circlesHarry, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title("Haris")
plt.xticks([])
plt.yticks([]) 

# Shi Tomasi algorithm

#copy of gray imaga
shiImgGray = grayImg.copy()

maxCorners = 9999
qualityLevel = 0.01
minDistance = 10
cornersShi = cv2.goodFeaturesToTrack(shiImgGray,maxCorners,qualityLevel,minDistance)

#draw circles 
imgShiTomasi = img.copy()

for i in cornersShi:  
    x,y = i.ravel()  
    cv2.circle(imgShiTomasi,(x,y),3,(B, G, R),-1)

#plot
plt.subplot(nrows, ncols,5),plt.imshow(cv2.cvtColor(imgShiTomasi, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title("Shi Tomasi")
plt.xticks([])
plt.yticks([]) 

# SIFT to detect 
siftGray = grayImg.copy()
imgSift = img.copy()
#Initiate SIFT detector 
##limit cornes to 50
sift = cv2.SIFT(50) 
kp = sift.detect(siftGray,None) 
#Draw keypoints  
imgSift = cv2.drawKeypoints(imgSift,kp,color=(B, G, R), flags = 4) 

#plot
plt.subplot(nrows, ncols,6),plt.imshow(cv2.cvtColor(imgSift, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title("Sift")
plt.xticks([])
plt.yticks([]) 

##brute force for 2 images
img1 = cv2.imread('GMIT1.jpg',0)
img2 = cv2.imread('GMIT2.jpg',0)


# Initiate SIFT detector
orb = cv2.ORB()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
#img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], flags=2)
img3 = drawMatches(img1,kp1,img2,kp2,matches[:40])

#add img to plot
plt.subplot(nrows, ncols,7),plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title("Sift")
plt.xticks([])
plt.yticks([])

##FLANN
imgFlann1 = cv2.imread('GMIT1.jpg',0)          
imgFlann2 = cv2.imread('GMIT2.jpg',0) 

# Initiate SIFT detector
siftFlann = cv2.SIFT()

# find the keypoints and descriptors with SIFT
kpFlann1, desFlann1 = sift.detectAndCompute(imgFlann1,None)
kpFlann2, desFlann2 = sift.detectAndCompute(imgFlann2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches1 = flann.knnMatch(desFlann1,desFlann2,k=2)



# Need to draw only good matches, so create a mask
matchesMask1 = [[0,0] for i in xrange(len(matches1))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches1):
    if m.distance < 0.7*n.distance:
        matchesMask1[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask1,
                   flags = 0)

matches1D = []
count = 0
for i in range(len(matchesMask1)):   
    for j in range(len(matchesMask1[i])):   
        matches1D.append(m)
        
##for m in matches1:
    ##matches1D.append(m)


    
#fran not working need python 3.
#imgFlann3 = cv2.drawMatchesKnn(imgFlann1,kpFlann1,imgFlann2,kpFlann2,matches,None,**draw_params)
imgFlann3 = drawMatches(imgFlann1,kpFlann1,imgFlann2,kpFlann2,matches1D)

#plot
plt.subplot(nrows, ncols,8),plt.imshow(cv2.cvtColor(imgFlann3, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title("Flann")
plt.xticks([])
plt.yticks([])

#read image for separate in HSV
imgRGB = cv2.imread('GMIT1.jpg',0) 

#trabsfor to HSV
imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

##add to pplot
plt.subplot(nrows, ncols,8),plt.imshow(imgHSV, cmap = 'gray')
plt.title("HSV")
plt.xticks([])
plt.yticks([])

#separete image in chane;s
imgHSV1 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(imgHSV1)

#add H chanel
plt.subplot(nrows, ncols,9),plt.imshow(h)
plt.title("HSV H")
plt.xticks([])
plt.yticks([])

#add S chanel
plt.subplot(nrows, ncols,10),plt.imshow(s, cmap = 'gray')
plt.title("HSV S")
plt.xticks([])
plt.yticks([])

#add v chanel
plt.subplot(nrows, ncols,11),plt.imshow(v, cmap = 'gray')
plt.title("HSV V")
plt.xticks([])
plt.yticks([])

#show and wait 
plt.show()
cv2.waitKey(0)


