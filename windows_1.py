#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2
import os
from perspective import pipeline, perspective_matrix, perspective

TEST_IMG_DIR = 'test_images'
OUTPUT_DIR = 'output_images'
IMG_SHAPE = (720, 1280, 3)

import numpy as np
import cv2
import matplotlib.pyplot as plt


def find_lines(img):
    # Begin with a histogram of the source image
    histogram = np.sum(img[img.shape[0]/2:,:], axis=0)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((img, img, img))*255
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9

    # Set height of windows
    window_height = np.int(img.shape[0]/nwindows)
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Set the width of the windows +/- margin
    margin = 100
    
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Paired fit
    #shift = (sum(rightx[0:500])-sum(leftx[0:500]))//500
    #pairedx = np.append(leftx, rightx-shift)
    #pairedy = np.append(lefty, righty)
    #fit = np.polyfit(pairedy, pairedx, 2)
    #left_fit = fit
    #right_fit = np.copy(fit)
    #right_fit[2] += shift
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Apply some coloration
    #out_img[lefty, leftx] = [255, 0, 0]
    #out_img[righty, rightx] = [0, 0, 255]
    #pairedx = np.union1d(leftx, np.maximum(rightx-shift,0))
    #pairedy = np.union1d(lefty, righty)
    #out_img[pairedy, pairedx] = [0, 255, 0]
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    
    return (out_img, left_fit, right_fit)


def process_image(img):
    
    M, Minv = perspective_matrix()
    undist, thresh, warped = pipeline(img, None, M)
    out_img, left_fit, right_fit = find_lines(warped)
    
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    #right_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + right_fit[2]-175

    detected = np.zeros_like(img)
    pts_left = np.array([x for x in zip(left_fitx, ploty)], np.int32)
    pts_right = np.array([x for x in zip(right_fitx, ploty)], np.int32)
    cv2.polylines(detected, [pts_left, pts_right], False, (255,255,0), 20)
    
    reshaped = perspective(detected, Minv)
    result = cv2.addWeighted(img, 0.8, reshaped, 1, 0)

    return result

if __name__ == '__main__':
    fname = os.path.join(TEST_IMG_DIR, 'test5.jpg')
    img = cv2.imread(fname)
    
    M, Minv = perspective_matrix()
    undist, thresh, warped = pipeline(img, None, M)
    out_img, left_fit, right_fit = find_lines(warped)

    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    #right_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + right_fit[2]-175

    detected = np.zeros_like(img)
    pts_left = np.array([x for x in zip(left_fitx, ploty)], np.int32)
    pts_right = np.array([x for x in zip(right_fitx, ploty)], np.int32)
    cv2.polylines(detected, [pts_left, pts_right], False, (255,255,0), 20)
    #cv2.polylines(detected, [pts_left], False, (255,255,0), 20)
    
    #reshaped = perspective(detected, Minv)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #result = cv2.addWeighted(img, 0.8, reshaped, 1, 0)
    #print(left_fitx)
    result = detected

    warped = np.dstack([warped,warped,warped])*255
    #result = cv2.addWeighted(warped, 0.8, detected, 1, 0)
    result = cv2.addWeighted(out_img, 0.8, detected, 1, 0)
    
    #plt.imshow(warped, cmap='gray')
    #plt.imshow(detected, cmap='gray')
    #plt.imshow(out_img)
    plt.imshow(result)
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.show()
