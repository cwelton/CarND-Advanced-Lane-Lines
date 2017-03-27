#!/usr/bin/env python
import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

CALIBRATION_DIR = 'camera_cal'
TEST_IMG_DIR = 'test_images'
OUTPUT_DIR = 'output_images'
IMG_SHAPE = (720, 1280, 3)

def _objPoints3D(points):
    '''Calculates a grid of points with dimensions provided.

    Returns an array of points like (0,0,0), (1,0,0), (2,0,0) ....,(9,6,0)
    '''
    objp = np.zeros((points[0]*points[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:points[0],0:points[1]].T.reshape(-1,2)
    return objp

def calibrate(images=None, points=(9,6)):
    '''Calibrates a camera matrix based on a set of test checkerboard images.'''
    
    assert(len(points) == 2)
    assert(min(points) > 0)

    if images is None:
        images = sorted(glob.glob(os.path.join(CALIBRATION_DIR, '*.jpg')))
        
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    objp = _objPoints3D(points)

    for fname in images:
        img = cv2.imread(fname)
        if img.shape != IMG_SHAPE:
            img = cv2.resize(img, (IMG_SHAPE[1], IMG_SHAPE[0]))
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, points, None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

    return cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

def undistort(img, calibration):
    '''Undistort a single image based on a camera calibration'''

    # Some of the sample images have an extra pixel, this can mess some subtle
    # things up, so fix it first.
    if img.shape != IMG_SHAPE:
        img = cv2.resize(img, (IMG_SHAPE[1], IMG_SHAPE[0]))

    # If we have no calibration results, return the image as given
    if calibration is None:
        return img

    # Otherwise run it through the undistortion process
    ret, mtx, dist, rvecs, tvecs = calibration
    return cv2.undistort(img, mtx, dist, None, mtx)


def undistort_examples(images, calibration):
    '''Undistorts a set of images based on a camera calibration.

    Returns a single image of the original and undistorted images stacked side
    by side, with each example image stacked vertically.
    '''
    ret, mtx, dist, rvecs, tvecs = calibration
    num_img = len(images)
    result = None
    for i, fname in enumerate(images):
        img = cv2.imread(images[i])
        dst = undistort(img, calibration)
        sidebyside = np.hstack((img,dst))
        if result is None:
            result = sidebyside
        else:
            result = np.concatenate((result, sidebyside))

    return result

if __name__ == '__main__':
    outfile1 = os.path.join(OUTPUT_DIR, 'calibration.jpg')
    outfile2 = os.path.join(OUTPUT_DIR, 'test-images.jpg')
    images = sorted(glob.glob(os.path.join(CALIBRATION_DIR, '*.jpg')))

    print("Calibrating...")
    calibration = calibrate(images)
    comparison = undistort_examples(images[0:4], calibration)
    cv2.imwrite(outfile1, comparison)
    print("calibration output written to {}".format(outfile1))

    images = sorted(glob.glob(os.path.join(TEST_IMG_DIR, '*.jpg')))
    comparison = undistort_examples(images, calibration)
    cv2.imwrite(outfile2, comparison)
    print("calibration output written to {}".format(outfile2))
