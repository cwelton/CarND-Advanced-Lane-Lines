#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2
import os
import sys
from perspective import pipeline, perspective_matrix, perspective, DST
from calibration import calibrate
from curvature import curvature

TEST_IMG_DIR = 'test_images'
OUTPUT_DIR = 'output_images'
IMG_SHAPE = (720, 1280, 3)

# window settings
WINDOW_WIDTH = 30
WINDOW_HEIGHT = IMG_SHAPE[0]/10
WINDOW_MARGIN = 80

class Cluster(object):

    def __init__(self, item=None, strength=None):
        if item is None:
            self.points = []
            self.strength = []
            self.mean = None
            self.std = None
            self.magnitude = None
        else:
            self.points = [item]
            self.strength = [strength]
            self.mean = None
            self.std = None
            self.magnitude = None

    def __str__(self):
        return str([self.points, self.strength])

    def __repr__(self):
        return str(self.center())

    def __len__(self):
        return len(self.points)
    
    def append(self, item):
        assert len(item) == 2
        self.points.append(item[0])
        self.strength.append(item[1])
        self.mean = None
        self.std = None
        self.magnitude = None
            
    def center(self):
        if self.mean is None or self.std is None:
            self.mean = np.mean(self.points)
            self.std = np.std(self.points)
            self.magnitude = max(self.strength)
            
        return (self.mean, self.std, self.magnitude)



def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),
           max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def find_clusters(choices):
    '''Identifies ranges of proximate points and groups them into clusters'''

    # Loop through the set of choices
    max_distance = 5
    clusters = []
    last = -max_distance
    for item in choices:

        # If the distance between the last point and the current point is less than
        # our threshold then add it to the current cluster, otherwise start a new
        # cluster
        if item[0]-last < max_distance:
            clusters[-1].append(item)
        else:
            clusters.append(Cluster(item[0], item[1]))

        last = item[0]

    # finally reduce each cluster to its mean and stddev and return the result
    return clusters


def find_centers(l_conv, r_conv, l_offset, r_offset, l_historic = [], r_historic = []):
    offset = WINDOW_WIDTH/2
    min_threshold = 200
    l_threshold = min_threshold #max([min_threshold,max(l_conv) * 0.8])
    r_threshold = min_threshold #max([min_threshold,max(r_conv) * 0.8])

    # Select the values over threshold and normalize to absolute positions in window
    l_choices = [(i + l_offset - offset, l_conv[i]) for (i,z) in enumerate(l_conv >= l_threshold) if z]
    r_choices = [(i + r_offset - offset, r_conv[i]) for (i,z) in enumerate(r_conv >= r_threshold) if z]
    l_clusters = find_clusters(l_choices)
    r_clusters = find_clusters(r_choices)

    width_goal = DST[1][0]-DST[0][0]

    # If no detected clusters, fall back on last known center
    if len(l_clusters) == 0 and len(l_historic) > 0:
        l_clusters = [Cluster(l_historic[0].mean, 0)]
    if len(r_clusters) == 0 and len(r_historic) > 0:
        r_clusters = [Cluster(r_historic[0].mean, 0)]


    # Summaries ensures that the mean values are filled out
    summaries = [x.center() for x in l_clusters+r_clusters]        
    
    distance_matrix = [
        [abs(right.mean - left.mean - width_goal),  # lane width fit
         left,
         right]
            for left in l_clusters for right in r_clusters]

    distance_matrix = sorted(distance_matrix, key=lambda x: x[0])
        
    # Select the minimum distance combination
    (distance, left, right) = distance_matrix[0]

    return (left, right)

def find_window_centroids(img,
                          window_width=WINDOW_WIDTH,
                          window_height=WINDOW_HEIGHT,
                          margin=WINDOW_MARGIN):

    # Store the (left,right) window centroid positions per level
    left_centroids = []
    right_centroids = []

    # Create our window template that we will use for convolutions
    window = np.ones(window_width) 
    
    # First find the two starting positions for the left and right lane by
    # using np.sum to get the vertical image slice and then np.convolve the
    # vertical image slice with the window template Sum quarter bottom of
    # image to get slice, could use a different ratio
    y_quartiles = [y*img.shape[0]//4 for y in range(5)]
    x_deciles   = [x*img.shape[1]//10 for x in range(11)]

    l_sum = np.sum(img[y_quartiles[3]:y_quartiles[4], x_deciles[1]:x_deciles[4]], axis=0)
    r_sum = np.sum(img[y_quartiles[3]:y_quartiles[4], x_deciles[6]:x_deciles[9]], axis=0)
    l_conv = np.convolve(window, l_sum)
    r_conv = np.convolve(window, r_sum)

    l_center, r_center = find_centers(l_conv, r_conv, x_deciles[1], x_deciles[6])

    # Add what we found for the first layer
    left_centroids.append(l_center)
    right_centroids.append(r_center)
    
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(img.shape[0]/window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(img[int(img.shape[0]-(level+1)*window_height):int(img.shape[0]-level*window_height),:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at
        # right side of window, not center of window
        offset = window_width/2
        l_min_index = int(max(l_center.mean+offset-margin,0))
        l_max_index = int(min(l_center.mean+offset+margin,img.shape[1]))
        r_min_index = int(max(r_center.mean+offset-margin,0))
        r_max_index = int(min(r_center.mean+offset+margin,img.shape[1]))
        
        l_conv = np.array(conv_signal[l_min_index:l_max_index])
        r_conv = np.array(conv_signal[r_min_index:r_max_index])
        l_center, r_center = find_centers(l_conv, r_conv, l_min_index, r_min_index,
                                          [l_center], [r_center])

        # Add what we found for that layer
        left_centroids.append(l_center)
        right_centroids.append(r_center)

    return (left_centroids, right_centroids)

def find_line(img):
    nonzero = img.nonzero()
    zy = np.array(nonzero[0])
    zx = np.array(nonzero[1])
    return np.polyfit(zy, zx, 2)

def find_window(img, left_centroids, right_centroids, window_height=WINDOW_HEIGHT):

    # if no window centers found, just return original image
    if len(left_centroids) == 0 or len(right_centroids) == 0:
        return img

    # Points used to draw all the left and right windows
    indicators = np.zeros_like(img)
    l_points = np.zeros_like(img)
    r_points = np.zeros_like(img)

    # Mask left and right lines to the points in the vicinity of the centroids
    left = np.array([(x.mean, img.shape[0]-i*window_height) for (i,x) in enumerate(left_centroids)], np.int32)
    right = np.array([(x.mean, img.shape[0]-i*window_height) for (i,x) in enumerate(right_centroids)], np.int32)
    cv2.polylines(l_points, [left], False, 1, 120)
    cv2.polylines(r_points, [right], False, 1, 120)
    
    l_points = cv2.bitwise_and(img, l_points)
    r_points = cv2.bitwise_and(img, r_points)
    left_fit = find_line(l_points)
    right_fit = find_line(r_points)

    for (i,c) in enumerate(left_centroids):
        c1 = (int(c.mean), int(img.shape[0]-i*window_height))
        cv2.circle(indicators, c1, 50, 1, 3)

    for (i,c) in enumerate(right_centroids):        
        c2 = (int(c.mean), int(img.shape[0]-i*window_height))
        cv2.circle(indicators, c2, 50, 1, 3)
        
    
    # Use the results to mask the image    
    masked = cv2.bitwise_or(l_points, r_points)*255
    return (masked, indicators, left_fit, right_fit)


FRAME = 0
def process_image(img, calibration=None, write_output=False):
    global FRAME
    
    M, Minv = perspective_matrix()
    _, _, warped = pipeline(img, calibration, M)

    left_centroids, right_centroids = find_window_centroids(warped)
    masked, indicators, left_fit, right_fit = find_window(warped, left_centroids, right_centroids)

    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    detected_left  = np.zeros_like(masked)
    detected_right = np.zeros_like(masked)    
    pts_left = np.array([x for x in zip(left_fitx, ploty)], np.int32)
    pts_right = np.array([x for x in zip(right_fitx, ploty)], np.int32)
    cv2.polylines(detected_left, [pts_left], False, 1, 25)
    cv2.polylines(detected_right, [pts_right], False, 1, 25)
    detected = cv2.bitwise_or(detected_left, detected_right)

    paved = np.zeros_like(detected)    
    pts = np.array([np.concatenate((pts_left,pts_right))])
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(paved, np.int_([pts]), 1)

    #pts_left = np.array([x for x in zip(left_fitx, ploty)], np.int32)
    #pts_right = np.array([x for x in zip(right_fitx, ploty)], np.int32)    
    #cv2.polylines(detected, [pts_left, pts_right], False, 1, 25)    

    # Make the minimap showing the processed line detection
    zeros = np.zeros_like(masked)
    masked_inv = cv2.bitwise_not(masked)
    color_masked = np.dstack([zeros,zeros,masked])
    color_indicated = np.dstack([indicators,zeros,zeros])*255
    color_detected = np.dstack([detected, detected, detected])*255
    color_paved = np.dstack([zeros, paved, zeros])*255
    color_markup = cv2.addWeighted(color_detected, 0.8, color_indicated, 0.8, 0)
    color_markup_masked = cv2.bitwise_and(color_markup, color_markup, mask=masked_inv)
    minimap_full = cv2.bitwise_or(color_markup_masked, color_masked)
    minimap = cv2.resize(minimap_full, (0,0), fx=0.25, fy=0.25)

    #plt.subplot(2,2,1)
    #plt.imshow(warped*255, cmap='gray')
    #plt.subplot(2,2,2)
    #plt.imshow(color_masked)
    #plt.subplot(2,2,3)
    #plt.imshow(color_indicated)
    #plt.subplot(2,2,4)
    #plt.imshow(color_detected)
    #plt.show()
    
    reshaped = perspective(color_paved, Minv)
    result = cv2.addWeighted(img, 1, reshaped, 0.3, 0)

    result[0:minimap.shape[0],0:minimap.shape[1]] = minimap

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Add a frame indicator
    cv2.putText(result, 'frame   %0.4d' % FRAME,
                (img.shape[1]-210, 40), font, 0.8,
                (255,255,255), 2, cv2.LINE_AA)
    
    # Calculate curvature
    (curve_left, curve_right, offset_from_center) = curvature(detected_left, detected_right)
    curve_display = np.mean([curve_left,curve_right])
    cv2.putText(result, 'curvature %6dm' % int(curve_display),
                (img.shape[1]-255, 70), font, 0.8,
                (255,255,255), 2, cv2.LINE_AA)

    # Display offset from center
    cv2.putText(result, 'center  %+3.2fm' % offset_from_center,
                (img.shape[1]-220, 100), font, 0.8,
                (255,255,255), 2, cv2.LINE_AA)
    
    if write_output:
        stacked = np.hstack([img,result])
        if not os.path.isfile('temp/%0.4d_original.jpg'):
            cv2.imwrite('temp/%0.4d_original.jpg' % FRAME, img)

        cv2.imwrite('temp/%0.4d_result.jpg' % FRAME, result)

    FRAME += 1
    return result

if __name__ == '__main__':

    calibration = calibrate()
    
    files = glob.glob(os.path.join(TEST_IMG_DIR,'*.jpg'))
    #files = [os.path.join(TEST_IMG_DIR, 'straight_lines1.jpg')]
    #files = [os.path.join(TEST_IMG_DIR, '0566_original.jpg')]
    #files = files[-4:-3]
    #files = files[-2:-1]
    #files = files[0:1]
    
    for fname in files:
        print(fname) 
        img = cv2.imread(fname)
        result = process_image(img, calibration, write_output=False)
        plt.imshow(result)
        plt.show()

