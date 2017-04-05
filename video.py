import pickle

from moviepy.editor import VideoFileClip
import calibration
from windows_3 import *

class Line(object):
    def __init__(self, shape, history_size = 10):        

        # Shape of expected images
        self.shape = shape
        
        # How many history items to keep
        self.history_size = history_size

        # Linear space in the y coordinate
        self.ploty = np.linspace(0, self.shape[0]-1, self.shape[0] )
        
        # was the line detected in the last iteration?
        self.detected = False

        # x values of the last n fits of the line
        self.recent_xfitted = []

        # centroids of the last n iterations
        self.recent_centroids = []
        
        #average x values of the fitted line over the last n iterations
        #self.bestx = None     

        #polynomial coefficients averaged over the last n iterations
        #self.best_fit = None  

        #polynomial coefficients for the most recent fit
        #self.current_fit = [np.array([False])]  

        #radius of curvature of the line in some units
        #self.radius_of_curvature = None
        self.recent_curvature = []
        self.recent_center = []
        
        #distance in meters of vehicle center from the line
        #self.line_base_pos = None 

        #difference in fit coefficients between last and new fits
        #self.diffs = np.array([0,0,0], dtype='float') 

        #x values for detected line pixels
        #self.allx = None  

        #y values for detected line pixels
        #self.ally = None

    def trim_history(self):
        if len(self.recent_xfitted) > self.history_size:
            self.recent_xfitted.pop(0)
        if len(self.recent_centroids) > self.history_size:
            self.recent_centroids.pop(0)
        if len(self.recent_curvature) > self.history_size:
            self.recent_curvature.pop(0)
        if len(self.recent_center) > self.history_size:
            self.recent_center.pop(0)            

    def record_fit(self, fit, centroids):
        self.recent_xfitted.append(fit)
        self.recent_centroids.append(centroids)
        self.detected = True  # FIXME
        self.trim_history()

    def record_curvature(self, curve):
        self.recent_curvature.append(curve)
        self.trim_history()

    def record_center(self, center):
        self.recent_center.append(center)
        self.trim_history()
        
    def points(self):
        fit = np.mean(self.recent_xfitted, axis=0)
        fitx = fit[0]*self.ploty**2 + fit[1]*self.ploty + fit[2]
        pts = np.array([x for x in zip(fitx, self.ploty)], np.int32)
        return pts

class VideoContext(object):

    def __init__(self, clipname):
        self.clip_in = VideoFileClip("project_video.mp4")
        self.shape = self.clip_in.size
        self.frame = 0
        self.recent_images = []
        self.left_line = Line(self.shape)
        self.right_line = Line(self.shape)
        self.calibration = calibration.calibrate()
        self.M, self.Minv = perspective_matrix()
        
    def save(self, outname):
        self.clip = self.clip_in.fl_image(self.next_image)
        self.clip.write_videofile(outname, audio=False)

    def next_image(self, img):
        self.recent_images.append(img)
        if len(self.recent_images) > 5:
            self.recent_images.pop(0)
            
        return self.process_image(img)

    def build_minimap(self, masked, indicators, detected):
        # Make the minimap showing the processed line detection
        zeros = np.zeros_like(masked)
        masked_inv = cv2.bitwise_not(masked)
        color_masked = np.dstack([zeros,zeros,masked])
        color_indicated = np.dstack([indicators,zeros,zeros])*255
        color_markup = cv2.addWeighted(detected, 0.8, color_indicated, 0.8, 0)
        color_markup_masked = cv2.bitwise_and(color_markup, color_markup, mask=masked_inv)
        minimap = cv2.bitwise_or(color_markup_masked, color_masked)
        return cv2.resize(minimap, (0,0), fx=0.25, fy=0.25)

    def adjust_window(self, img):
        # TODO: move these to be config not globals
        window_height = WINDOW_HEIGHT
        window_width = WINDOW_WIDTH
        margin = WINDOW_MARGIN/2

        left_centroids = self.left_line.recent_centroids[-1]
        right_centroids = self.right_line.recent_centroids[-1]

        window = np.ones(window_width) 

        # Go through each layer looking for max pixel locations
        for level in range(0,(int)(img.shape[0]/window_height)):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(img[int(img.shape[0]-(level+1)*window_height):int(img.shape[0]-level*window_height),:], axis=0)
            conv_signal = np.convolve(window, image_layer)
            
            offset = window_width/2
            target = [level, level-1][level > 0]
            l_min_index = int(max(left_centroids[target].mean+offset-margin,0))
            l_max_index = int(min(left_centroids[target].mean+offset+margin,img.shape[1]))
            r_min_index = int(max(right_centroids[target].mean+offset-margin,0))
            r_max_index = int(min(right_centroids[target].mean+offset+margin,img.shape[1]))

            l_conv = np.array(conv_signal[l_min_index:l_max_index])
            r_conv = np.array(conv_signal[r_min_index:r_max_index])
            l_center, r_center = find_centers(l_conv, r_conv, l_min_index, r_min_index,
                                              [left_centroids[level]], [right_centroids[level]])

            # If one fit is significantly better than the other then simply adjust by lane width
            if l_center.magnitude > 2*r_center.magnitude:
                r_center.mean = l_center.mean+495
            if r_center.magnitude > 2*l_center.magnitude:
                l_center.mean = r_center.mean-495
            
            left_centroids[level] = l_center
            right_centroids[level] = r_center
        
        (masked, indicators, left_fit,
         right_fit) = find_window(img, left_centroids, right_centroids)
        return (masked, indicators, left_fit, right_fit, left_centroids, right_centroids)
    
    
    def process_image(self, img):
        _, _, warped = pipeline(img, self.calibration, self.M)

        if not (self.left_line.detected and self.right_line.detected):
            left_centroids, right_centroids = find_window_centroids(warped)
            (masked, indicators, left_fit,
             right_fit) = find_window(warped, left_centroids, right_centroids)
        else:
            (masked, indicators, left_fit, right_fit,
             left_centroids, right_centroids) = self.adjust_window(warped)

        self.left_line.record_fit(left_fit, left_centroids)
        self.right_line.record_fit(right_fit, right_centroids)

        pts_left = self.left_line.points()
        pts_right = self.right_line.points()
        
        detected_left = np.zeros_like(masked)
        detected_right = np.zeros_like(masked)
        cv2.polylines(detected_left, [pts_left], False, 1, 25)
        cv2.polylines(detected_right, [pts_right], False, 1, 25)
        detected = cv2.bitwise_or(detected_left, detected_right)
        
        # Todo, handle this more cleanly, very redundant with polylines above
        zeros = np.zeros_like(masked)        
        paved = np.zeros_like(detected)
        ploty = self.left_line.ploty;
        left_fitx =  fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx =  fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(paved, np.int_([pts]), 1)
        
        color_detected = np.dstack([detected, detected, detected])*255
        #reshaped = perspective(color_detected, self.Minv)
        #result = cv2.addWeighted(img, 0.8, reshaped, 1, 0)

        color_paved = np.dstack([zeros, paved, zeros])*255
        reshaped = perspective(color_paved, self.Minv)        
        result = cv2.addWeighted(img, 0.8, reshaped, 0.3, 0)

        # Place the minimap in the upper left corner
        minimap = self.build_minimap(masked, indicators, color_detected)
        result[0:minimap.shape[0],0:minimap.shape[1]] = minimap

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Add a frame indicator
        cv2.putText(result, 'frame   %0.4d' % self.frame,
                    (img.shape[1]-210, 40), font, 0.8,
                    (255,255,255), 2, cv2.LINE_AA)
        
        # Calculate the running average of curvature
        (curve_left, curve_right, offset_from_center) = curvature(detected_left, detected_right)
        if curve_left < 600:
            curve_left = np.mean(self.left_line.recent_curvature)
        if curve_right < 600:
            curve_right = np.mean(self.right_line.recent_curvature)

        self.left_line.record_curvature(curve_left)
        self.left_line.record_center(offset_from_center)
        self.right_line.record_curvature(curve_right)
        curve = np.mean(np.concatenate([self.left_line.recent_curvature,
                                        self.right_line.recent_curvature]))
        if curve > 9999:
            curve_display = ' >9999m'
        else:
            curve_display = '%6dm' % int(curve)
            cv2.putText(result, 'curvature %s' % curve_display,
                        (img.shape[1]-255, 70), font, 0.8,
                        (255,255,255), 2, cv2.LINE_AA)

        # Display offset from center
        mean_center = np.mean(self.left_line.recent_center)
        cv2.putText(result, 'center  %+3.2fm' % mean_center,
                    (img.shape[1]-220, 100), font, 0.8,
                    (255,255,255), 2, cv2.LINE_AA)
        
        self.frame += 1
        return result

if __name__ == '__main__':
    v = VideoContext("project_video.mp4")
    v.save("output.mp4")
    
