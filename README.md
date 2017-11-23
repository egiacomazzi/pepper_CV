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

