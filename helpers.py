
import numpy as np
import cv2

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0,255)):
    """Calculates directional gradient in either x or y axis"""
    
    assert(orient in 'xy')
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    dx, dy = [(0,1),(1,0)][orient == 'x']
    sobel = cv2.Sobel(gray, cv2.CV_64F, dx, dy, ksize=sobel_kernel)
    
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scale_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    # 5) Create a mask of 1's where the scaled gradient magnitude 
    #    is > thresh_min and < thresh_max
    binary = np.zeros_like(scale_sobel)
    binary[(scale_sobel >= thresh[0]) & (scale_sobel <= thresh[1])] = 1
    
    return binary

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    """Calculates the magnitude of gradient"""
    
    assert(sobel_kernel > 1 and sobel_kernel % 2 == 1)
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # 3) Calculate the magnitude 
    mag_sobel = np.sqrt(sobelx**2 + sobely**2)

    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scale_sobel = np.uint8(255*mag_sobel/np.max(mag_sobel))
    
    # 5) Create a binary mask where mag thresholds are met
    binary = np.zeros_like(scale_sobel)
    binary[(scale_sobel >= mag_thresh[0]) & (scale_sobel <= mag_thresh[1])] = 1

    # 6) Return this mask as your binary_output image
    return binary

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    """Calculates the direction of gradient"""

    assert(sobel_kernel > 1 and sobel_kernel % 2 == 1)
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    sobel_dir = np.arctan2(abs_sobely, abs_sobelx)
    
    # 5) Create a binary mask where mag thresholds are met
    binary = np.zeros_like(sobel_dir)
    binary[(sobel_dir >= thresh[0]) & (sobel_dir <= thresh[1])] = 1
    
    # 6) Return this mask as your binary_output image
    return binary
