#!/usr/bin/env python
import numpy as np

from perspective import DST

# Conversions from pixelspace to meters
#  - Lane width comes close to 3.7 meters
YM_PER_PIXEL = 30/350
XM_PER_PIXEL = 3.7/(DST[1][0]-DST[0][0])

def _formula(fit, yval):
    ''' R_curve = (1+(2Ay+B)^2)^1.5 / |2A| '''
    A,B = fit[0:2]
    return (1+(2*A*yval+B)**2)**1.5 / np.absolute(2*A)

def _calculate(points, yval):
    nonzero = points.nonzero()
    zy = np.array(nonzero[0])*YM_PER_PIXEL
    zx = np.array(nonzero[1])*XM_PER_PIXEL    
    fit = np.polyfit(zy, zx, 2)
    curve = _formula(fit, yval)
    position = fit[0]*yval**2 + fit[1]*yval + fit[2]
    return (curve, position)

def curvature(left, right):

    # Y point to evaluate the curvature
    assert left.shape[0] == right.shape[0]
    maxy = left.shape[0]*YM_PER_PIXEL
    midx = left.shape[1]*XM_PER_PIXEL/2

    # Calculate the curvature for each line
    left_curve_rad, left_position = _calculate(left, maxy)
    right_curve_rad, right_position = _calculate(right, maxy)

    # Calculate offset from center
    estimated_lane_width = right_position - left_position
    position = left_position + estimated_lane_width/2.0
    offset_from_center = midx - position
    return (left_curve_rad, right_curve_rad, offset_from_center)
