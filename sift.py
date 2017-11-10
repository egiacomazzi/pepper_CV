import numpy as np
import cv2

def only_white(img):
'''
    removes all gray values only white part of the picture doesn't become black
'''
    m,n = img.shape
    for x in range(m):
        for y in range(n):
            if img[x,y]<210:
                img[x,y]= 0
            else:
                img[x,y]= img[x,y]
    return cv2.imwrite('img_white_only.jpg',img)

def get_deskriptors1(sift,img):
    '''
        Takes image and returns deskriptors and saves a version of the image with deskriptors drawn in
    '''

    # kp_house = sift.detect(img,None)
    # kp_org = sift.detect(img_original,None)

    orb = cv2.ORB_create()
    _, des = orb.detectAndCompute(img,None)
    #_, des_house = orb.detectAndCompute(img_house,None)

    return des

    #v2.imwrite('house_keypoints.jpg',img2)
def draw_deskriotors(des,img):
    img_with_des = cv2.drawKeypoints(img,des,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('keypoints.jpg',img)

def matching(img1,img2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_org,des_house)
    matches = sorted(matches, key = lambda x:x.distance)
    img3 = cv2.drawMatches(img_original,kp_org,img_house,kp_house,matches[:],img_original, flags=2)
    cv2.imwrite('matching_img.jpg',img3)


_, des_org1 = sift.compute(img_original,kp_org)
_, des_house1 = sift.compute(img_house,kp_house)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des_org1,des_house1, k=2)
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

img4 = cv2.drawMatchesKnn(img_original,kp_org,img_house,kp_house,good,img_original,flags=2)
cv2.imwrite('matching_img2.jpg',img4)

#surf
# _, des_org1 = sift.compute(img_original,kp_org)
# _, des_house1 = sift.compute(img_house,kp_house)
#
# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des_org1,des_house1, k=2)
# good = []
# for m,n in matches:
#     if m.distance < 0.75*n.distance:
#         good.append([m])
#
# img4 = cv2.drawMatchesKnn(img_original,kp_org,img_house,kp_house,good,img_original,flags=2)
# cv2.imwrite('matching_img2.jpg',img4)

def __init__():
    #save images
    img_original = cv2.imread('img.png',0)          # queryImage
    img_house = cv2.imread('white_house.jpg',0) # trainImage

    sift = cv2.xfeatures2d.SIFT_create()
