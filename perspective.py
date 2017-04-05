#!/usr/bin/env python
import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from calibration import calibrate, undistort

TEST_IMG_DIR = 'test_images'
OUTPUT_DIR = 'output_images'
IMG_SHAPE = (720, 1280, 3)

#SRC = np.array([(260,672),(1050,672),(710,464),(572,464)], 'float32')
#SRC = np.array([(260,672),(1050,672),(708,464),(574,464)], 'float32')
#DST = np.array([(260,672),(1050,672),(1050,464),(260,464)], 'float32')

SRC = np.array([(265,677),(1040,677),(677,443),(604,443)], 'float32')
DST = np.array([(405,710),(900,710),(900,-100),(405,-100)], 'float32')

def perspective_matrix(src=SRC, dst=DST):
    '''Returns perspective and inverse perspective matrices'''
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return (M, Minv)

def perspective(img, M):
    '''Applies a perspective transform to an image'''
    (h,w) = img.shape[0:2]
    return cv2.warpPerspective(img, M, (w,h))


def skeletonize(img):
    """ OpenCV function to return a skeletonized version of img, a Mat object"""

    #  hat tip to http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/

    img = img.copy() # don't clobber original
    skel = img.copy()

    skel[:,:] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    while True:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp  = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img[:,:] = eroded[:,:]
        if cv2.countNonZero(img) == 0:
            break

    return skel

#def color_threshold(img, sx_thresh=(20,100), s_thresh=(170,255)):
#def color_threshold(img, sx_thresh=(18,100), s_thresh=(175,235)):

def color_threshold(img, sx_thresh=(30,100), s_thresh=(140,235)):    

    # Convert to HSL color space and separate the V channel
    hsl = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsl[:,:,1]
    s_channel = hsl[:,:,2]

    l_blurred = cv2.blur(l_channel, (5,5))    
    
    # Sobel x
    sobelx = cv2.Sobel(l_blurred, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( np.zeros_like(sxbinary), s_binary, sxbinary))

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return combined_binary

def bin2Color(img):
    return np.dstack((img,img,img))*255

def pipeline(img, calibration, M, drawlines=False):
    undist = undistort(img, calibration)
    thresh = color_threshold(undist)
    warped = perspective(thresh, M)

    if drawlines:
        undist = cv2.polylines(undist, [poly], True, red, 3)
        thresh = bin2Color(thresh)
        cv2.polylines(thresh, [poly], True, red, 3)
        warped = bin2Color(warped)
        cv2.line(warped, tuple(DST[0]), tuple(DST[3]), red, 3)
        cv2.line(warped, tuple(DST[1]), tuple(DST[2]), red, 3)

    return (undist, thresh, warped)

if __name__ == '__main__':
    calibration = calibrate()

    outfile = os.path.join(OUTPUT_DIR, 'warped.jpg')        
    images = sorted(glob.glob(os.path.join(TEST_IMG_DIR,'*.jpg')))
    num_img = len(images)
    red = (0,0,255)
    
    poly = np.array(SRC, np.int32)
    poly = poly.reshape((-1,1,2))
    M, Minv = perspective_matrix()

    vstacked = None
    for i, fname in enumerate(images):
        #if fname.endswith("_original.jpg"):
        #    continue
        img = cv2.imread(fname)
        undist, thresh, warped = pipeline(img, calibration, M, True)
        stacked = np.hstack((undist,thresh,warped))
        if vstacked is None:
            vstacked = stacked
        else:
            vstacked = np.concatenate((vstacked, stacked))

    cv2.imwrite(outfile, vstacked)
    print("warp output written to {}".format(outfile))

