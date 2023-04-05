import numpy as np
import cv2 as cv
from picamera2 import Picamera2
from math import *

#my_file = open('image.png', 'wb')
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

def uTheta(u):
    return (-5e-11*(u**4)) + (6e-08*(u**3)) - (2e-05*(u**2)) - (0.0015*u) + (0.6468)

def sbD(sb):
    i = 1/sb
    return (1712.5*i) + (5.6904)

while True:
    frame = picam2.capture_array()
    
    # It converts the BGR color space of image to HSV color space
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    # Threshold of blue in HSV space
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    
    # preparing the mask to overlay
    mask = cv.inRange(hsv, lower_yellow, upper_yellow)
    
    result = cv.bitwise_and(frame, frame, mask = mask)
    gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)
        
    rows = gray.shape[0]
    #frame2 = picam2.capture_file("blob.png")
    #gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    params = cv.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 500
    params.maxArea = 100000
    params.filterByColor = False
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    
    
    # Set up the detector with default parameters.
    detector = cv.SimpleBlobDetector_create(params)
    X = 0
    Y = 0
     
    # Detect blobs.
    keypoints = detector.detect(gray)
    for keypoint in keypoints:
        theta = uTheta(keypoint.pt[0])
        d = sbD(keypoint.size)
        X = d * sin(theta) 
        Y = d * cos(theta)
    print('X:', X)
    print('Y:', Y)
    
    frame_with_keypoints = cv.drawKeypoints(result, keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
     
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    #im_with_keypoints = cv2.drawKeypoints(mask, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
     
    # Show keypoints
    cv.imshow("Keypoints", frame_with_keypoints)
    
    #cv.waitKey(0)
    #print('cv.waitKey(0)')
    
    # Display the resulting frame
    #cv2.imshow('frame',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cv.destroyAllWindows()
print('cv.destroyAllWindows()')