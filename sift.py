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
    return kp, des

def bfMatching_SIFT(bf, kp_img1, kp_img2, des_img1, des_img2, img1, img2):
    matches = bf.knnMatch(des_img1, des_img2, k=2)
    good = []
    good2 = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
            good2.append(m)

    img_match = cv2.drawMatchesKnn(img1, kp_img1, img2, kp_img2, good, img1, flags=2)
    cv2.imwrite('matching_SIFT_BFM.jpg',img_match)

    MIN_MATCH_COUNT = 10
    if len(good)>=MIN_MATCH_COUNT:
            src_pts = np.float32([ kp_img1[m.queryIdx].pt for m in good2 ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp_img2[m.trainIdx].pt for m in good2 ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()

            h,w, = img1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)

            img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                singlePointColor = None,
                matchesMask = matchesMask, # draw only inliers
                flags = 2)

    img3 = cv2.drawMatches(img1, kp_img1, img2, kp_img2, good2, None, **draw_params)
    cv2.imwrite('hologram.jpg',img3)
    #return good

############################### ORB ##############################################################
'''
    functions for getting keypoints and descriptors using ORB, Brute-Force Matcher(BFM) for ORB
'''
def get_key_des_ORB(orb, img):
    kp, des = orb.detectAndCompute(img,None)
    return kp, des

def matching_BFM_ORB(bf,kp_img1, kp_img2, des_img1, des_img2, img1, img2):
    matches = bf.match(des_img1,des_img2)
    matches = sorted(matches, key = lambda x:x.distance)
    img3 = cv2.drawMatches(img1,kp_img1,img2,kp_img2,matches[:],img1, flags=2)
    cv2.imwrite('matching_BFM_ORB.jpg',img3)
    #return matches

############################### Matching FLANN ###############################
'''
    matching for SIFT and ORB with FLANN matcher
'''

def matching_FLANN(flann, kp_img1, kp_img2, des_img1, des_img2, img1, img2):
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
    #return matches

def hologram(good,kp_img1, kp_img2, img1, img2):
    MIN_MATCH_COUNT = 10
    if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp_img1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp_img2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()

            h,w,d = img1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)

            img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                singlePointColor = None,
                matchesMask = matchesMask, # draw only inliers
                flags = 2)

    img3 = cv2.drawMatches(img1,kp_img1,img2,kp_img2,good,None,**draw_params)
    cv2.imwrite('hologram.jpg',img3)


############################### Draw Keypoints ###############################
def draw_keypoints(kp, img):
    img = cv2.drawKeypoints(img,kp,img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('keypoints.jpg',img)

def main():
    #creat objects
    sift = cv2.xfeatures2d.SIFT_create()
    orb = cv2.ORB_create()
    bf_sift = cv2.BFMatcher(cv2.NORM_L2)
    bf_ORB = cv2.BFMatcher(cv2.NORM_HAMMING)
    # FLANN parameters SIFT
    FLANN_INDEX_KDTREE = 1
    index_params_SIFT = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params_SIFT = dict(checks=50)   # or pass empty dictionary
    flann_SIFT = cv2.FlannBasedMatcher(index_params_SIFT,search_params_SIFT)
    # FLANN parameters ORB
    FLANN_INDEX_LSH = 6
    index_params_ORB = dict(algorithm = FLANN_INDEX_LSH,
                       table_number = 6, # 12,6
                       key_size = 12,     # 20,12
                       multi_probe_level = 1) #2,1
    search_params_ORB = dict(checks=100)  # or pass empty dictionary
    flann_ORB = cv2.FlannBasedMatcher(index_params_ORB,search_params_ORB)

    #import images
    nachttisch_tasse = cv2.imread('ricola_laptop.jpg',0)          # queryImage
    tasse = cv2.imread('smarti_tisch.jpg',0) # trainImage

    kp_tasse, des_tasse = get_descriptor_sift(sift, tasse)
    kp_nachttisch, des_nachttisch = get_descriptor_sift(sift, nachttisch_tasse)
    bfMatching_SIFT(bf_sift, kp_tasse, kp_nachttisch, des_tasse, des_nachttisch, tasse, nachttisch_tasse)


    # tasse = cv2.imread('tasse_c.JPG',0)
    # nachttisch_tasse = cv2.imread('nachttisch_tasse_c.JPG',0)
    # schreibtisch = cv2.imread('schreibtisch_c.JPG',0)
    #save_pic(schreibtisch)

    # detect tasse on nachttisch
    ###### SIFT BFM #####
    # kp_tasse, des_tasse = get_descriptor_sift(sift, tasse)
    # kp_nachttisch, des_nachttisch = get_descriptor_sift(sift, nachttisch_tasse)
    # bfMatching_SIFT(bf_sift, kp_tasse, kp_nachttisch, des_tasse, des_nachttisch, tasse, nachttisch_tasse)
    #hologram(good,kp_tasse, kp_nachttisch,tasse, nachttisch_tasse)
    ###### SIFT FLANN #####
    #matching_FLANN(flann_SIFT, kp_tasse, kp_nachttisch, des_tasse, des_nachttisch, tasse, nachttisch_tasse)

    #draw_keypoints(kp_tasse,tasse)
    #draw_keypoints(kp_nachttisch,nachttisch_tasse)

    ##### ORB BFM #####
    # kp_tasse, des_tasse = get_key_des_ORB(orb, tasse)
    # kp_nachttisch, des_nachttisch = get_key_des_ORB(orb, nachttisch_tasse)
    # matching_BFM_ORB(bf_ORB, kp_tasse, kp_nachttisch, des_tasse, des_nachttisch, tasse, nachttisch_tasse)
    #
    # ###### ORB FLANN #####
    # matching_FLANN(flann_ORB, kp_tasse, kp_nachttisch, des_tasse, des_nachttisch, tasse, nachttisch_tasse)


main()
