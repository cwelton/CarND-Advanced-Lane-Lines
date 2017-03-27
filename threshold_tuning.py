#!/usr/bin/env python
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Slider, Button, RadioButtons


from helpers import *

IMAGE = mpimg.imread('test_images/test5.jpg')

def make_combined(img, ksize, xthresh, ythresh, magthresh, dirthresh):
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=xthresh)
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=ythresh)
    mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=magthresh)
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=dirthresh)

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    return combined

# Edit this function to create your own pipeline.
def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
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
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    return color_binary




# Add controls
fig, ax = plt.subplots()
axes = [
    plt.axes([0.20, 0.17, 0.65, 0.03]),
    plt.axes([0.20, 0.12, 0.65, 0.03]),
    plt.axes([0.20, 0.07, 0.65, 0.03]),
    plt.axes([0.20, 0.02, 0.65, 0.03]),
]
x_min = Slider(axes[0], 'x_min', 0, 255, valinit=170, valfmt='%0.0f')
x_max = Slider(axes[1], 'x_max', 0, 255, valinit=255, valfmt='%0.0f')
y_min = Slider(axes[2], 'y_min', 0, 255, valinit=20, valfmt='%0.0f')
y_max = Slider(axes[3], 'y_max', 0, 255, valinit=100, valfmt='%0.0f')

def update(_):
    x_thresh = (x_min.val, x_max.val)
    y_thresh = (y_min.val, y_max.val)
    if False:
        combined = make_combined(IMAGE, 3, x_thresh, y_thresh,
                                 (20, 100),
                                 (0.7,1.3))
        ax.imshow(combined, cmap='gray')
    else:
        combined = pipeline(IMAGE, x_thresh, y_thresh)
        ax.imshow(combined, cmap='gray')

    fig.canvas.draw_idle()

x_min.on_changed(update)
x_max.on_changed(update)
y_min.on_changed(update)
y_max.on_changed(update)

update(1)

plt.show()


