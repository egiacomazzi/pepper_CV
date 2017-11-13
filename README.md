# Pepper's Computer Vision

## Basics ##

### 1. Feature detection ###

There are different ways to find features in pictures.
  
1. SIFT (Scale Invariant Feature Transform)  
	How does it work?  
	* read this [paper](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf) 
	* look at the [openCV documentation](https://docs.opencv.org/trunk/da/df5/tutorial_py_sift_intro.html)
	* or find other explanations  
	
	**but** SIFT is patented.

	
2. ORB (Oriented FAST and Rotated BRIEF)  
	How does it work?  
	* look at the [openCV documentation](https://docs.opencv.org/trunk/d1/d89/tutorial_py_orb.html)
	* or find other explanations    

> Find both possibilities implemented in my code.
	
### 2. Matching ###
There are two different methods for matching descriptors. They both work with SIFT descriptors as well as with ORB descriptors. They are explained [here](https://docs.opencv.org/trunk/dc/dc3/tutorial_py_matcher.html).  


> Find all possibilities implemented in my code.


At this point you can either print that the object was found or can draw a square around it with the hologram function.

## Code ##
I try to explaine my code for a faster understanding.
You will need openCV installed. Check to install it with the correct dependencies (Python2.7 or 3.0?) and do not give up. It almost took me 2 days to get this done.  
```python
import numpy as np
import cv2
```

#### Keypoints and Descriptors
SIFT
```python
def get_descriptor_sift(sift, img):
    kp, des = sift.detectAndCompute(img,None)
    return kp, des
```

ORB
```python
def get_key_des_ORB(orb, img):
    kp, des = orb.detectAndCompute(img,None)
    return kp, des
```

#### Matching

BFM with SIFT
```python
def bfMatching_SIFT(bf, kp_img1, kp_img2, des_img1, des_img2, img1, img2):
    #bf.knnMatch(des_img1, des_img2, k=2) : returns k best matches where k is specified by the user
    matches = bf.knnMatch(des_img1, des_img2, k=2)

    good = []
    global good2
    good2 = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
            good2.append(m)
    if (len(good) > 10):
         print("Found object in picture.")
         
    img_match = cv2.drawMatchesKnn(img1, kp_img1, img2, kp_img2, good, img1, flags=2)
    cv2.imwrite('matching_SIFT_BFM.jpg',img_match)
    return good, good2
```

BFM with ORB
```python
