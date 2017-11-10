import numpy as np
import cv2

'''
    Matching: https://docs.opencv.org/trunk/dc/dc3/tutorial_py_matcher.html
'''



def only_white(img):

    m,n = img.shape
    for x in range(m):
        for y in range(n):
            if img[x,y]<210:
                img[x,y]= 0
            else:
                img[x,y]= img[x,y]
    cv2.imwrite('img_white_only.jpg',img)

def save_pic(img):
    cv2.imwrite('new_pic.jpg',img)

######################## SIFT ##############################################################
'''
    functions for getting keypoints and descriptors using SIFT, Brute-Force Matcher(BFM) for SIFT
'''
def get_keypoints1(sift,img):
    return sift.detect(img,None)

def get_descriptor_sift(sift, img):
    kp, des = sift.detectAndCompute(img,None)
    return des

def bfMatching_SIFT(bf, des_img1, des_img2, img1, img2):
    matches = bf.knnMatch(des_img1, des_img2, k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    img_match = cv2.drawMatchesKnn(img1, kp_img1, img1, kp_img2, good, img1, flags=2)
    cv2.imwrite('matching_img2.jpg',img_match)
    return matches

############################### ORB ##############################################################
'''
    functions for getting keypoints and descriptors using ORB, Brute-Force Matcher(BFM) for ORB
'''
def get_key_des_ORB(orb, img):
    kp, des = orb.detectAndCompute(img,None)
    return kp, des

def matching_BFM_ORB(bf, des_img1, des_img2, img1, img2):
    matches = bf.match(des_img1,des_img2)
    matches = sorted(matches, key = lambda x:x.distance)
    img3 = cv2.drawMatches(img_1,kp_org,img_house,kp_house,matches[:10],img_original, flags=2)
    cv2.imwrite('matching_BFM_ORB.jpg',img3)
    return matches

############################### Matching FLANN ###############################
'''
    matching for SIFT and ORB with FLANN matcher
'''

def matching_FLANN(flann, des_img1, des_img2, img1, img2):
    matches = flann.knnMatch(des_img1,des_img2,k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in xrange(len(matches))]
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask,
                       flags = 0)
    img4 = cv2.drawMatchesKnn(img1,kp_img1,img2,kp_img2,matches,None,**draw_params)
    cv2.imwrite('matching_FLANN.jpg',img4)
    return matches
############################### Draw Keypoints ###############################
def draw_keypoints(kp, img):
    img_with_kp = cv2.drawKeypoints(img,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('keypoints.jpg',img)

def main():
    #creat objects
    sift = cv2.xfeatures2d.SIFT_create()
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # FLANN parameters SIFT
    FLANN_INDEX_KDTREE = 1
    index_params_SIFT = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params_SIFT = dict(checks=50)   # or pass empty dictionary
    flann_SIFT = cv2.FlannBasedMatcher(index_params_SIFT,search_params_SIFT)
    # FLANN parameters ORB
    FLANN_INDEX_LSH = 6
    search_params_ORB = dict(algorithm = FLANN_INDEX_LSH,
                       table_number = 6, # 12
                       key_size = 12,     # 20
                       multi_probe_level = 1) #2
    search_params_ORB = dict(checks=50)   # or pass empty dictionary
    flann_ORB = cv2.FlannBasedMatcher(search_params_ORB,search_params_ORB)

    #import images
    img_original = cv2.imread('img.png',0)          # queryImage
    img_house = cv2.imread('white_house.jpg',0) # trainImage

    tasse = cv2.imread('tasse_c.JPG',0)
    nachttisch_tasse = cv2.imread('nachttisch_tasse_c.JPG',0)
    schreibtisch = cv2.imread('schreibtisch_c.JPG',0)
    save_pic(schreibtisch)

    #kp, des = get_deskriptors1(sift,orb,img)
    #saves picture with keypoints
    #draw_keypoints(get_keypoints1(sift,img),img)
main()
