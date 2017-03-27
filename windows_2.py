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

# window settings
window_width = 50 
window_height = IMG_SHAPE[0]/9
margin = 100 

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),
           max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def find_window_centroids(img, window_width, window_height, margin):

    # Store the (left,right) window centroid positions per level
    window_centroids = []

    # Create our window template that we will use for convolutions
    window = np.ones(window_width) 
    
    # First find the two starting positions for the left and right lane by
    # using np.sum to get the vertical image slice and then np.convolve the
    # vertical image slice with the window template Sum quarter bottom of
    # image to get slice, could use a different ratio
    l_sum = np.sum(img[int(3*img.shape[0]/4):,:int(img.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(img[int(3*img.shape[0]/4):,int(img.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(img.shape[1]/2)
    
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(img.shape[0]/window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(img[int(img.shape[0]-(level+1)*window_height):int(img.shape[0]-level*window_height),:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at
        # right side of window, not center of window
        offset = window_width/2
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,img.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,0))
        r_max_index = int(min(r_center+offset+margin,img.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
        # Add what we found for that layer
        window_centroids.append((l_center,r_center))

    return window_centroids

def find_window(img):
    window_centroids = find_window_centroids(img, window_width, window_height, margin)

    # If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(img)
        r_points = np.zeros_like(img)

        # Go through each level and draw the windows    
        for level in range(0,len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width,window_height,img,window_centroids[level][0],level)
            r_mask = window_mask(window_width,window_height,img,window_centroids[level][1],level)

            # Add graphic points from window mask here to total pixels found 
            l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
            r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

        # Draw the results
        # add both left and right window pixels together
        template = np.array(r_points+l_points,np.uint8)

        # create a zero color channle 
        zero_channel = np.zeros_like(template)

        # make window pixels green
        template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8)

        # making the original road pixels 3 color channels
        warpage = np.array(cv2.merge((img,img,img)),np.uint8)

        # overlay the orignal road image with window results
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) 
        
    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((img,img,img)),np.uint8)

    return output

def process_image(img):
    M, Minv = perspective_matrix()
    undist, thresh, warped = pipeline(img, None, M)
    output = find_window(warped)
    reshaped = perspective(output, Minv)
    result = cv2.addWeighted(img, 0.8, reshaped, 1, 0)
    return result

if __name__ == '__main__':
    fname = os.path.join(TEST_IMG_DIR, 'test5.jpg')
    img = cv2.imread(fname)

    M, Minv = perspective_matrix()
    undist, thresh, warped = pipeline(img, None, M)
    output = find_window(warped)

    reshaped = perspective(output, Minv)

    result = cv2.addWeighted(img, 0.8, reshaped, 1, 0)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    plt.subplot(2,2,1)
    plt.imshow(warped, cmap='gray')
    plt.subplot(2,2,2)
    plt.imshow(output, cmap='gray')
    plt.subplot(2,2,3)
    plt.imshow(reshaped, cmap='gray')
    plt.subplot(2,2,4)
    plt.imshow(result)
    plt.show()
