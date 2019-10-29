import time
import cv2
import numpy as np
from sklearn.externals import joblib


""" load KNN classifier """
classifier = joblib.load('saved_model_3.pkl')
classes = {0:'Empty',1:'Up',2:'Down'}


def Vehicle_Detector(image):

    """ 1. image to grayscale & Gaussian filtering"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0) # Erase Noise

    """ 2. Edge Detection using Canny algorithm"""
    edged = cv2.Canny(blurred, 80, 200)  # Canny Edge Detection
    #cv2.imshow("edge", edged)

    kernel = np.ones((5,5),np.uint8)
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel) # Edge enhancement

    """ 3. find contours from edge image """
    _, contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #image = cv2.drawContours(image, contours, -1, (0, 0, 255), 3)
    #cv2.imshow("contour", image)

    for cnt in contours:

        """ ignore small contours """
        if not cv2.contourArea(cnt) > 1000:
            continue

        """ 4. get minAreaRect """
        rect = cv2.minAreaRect(cnt)  # Rotated Rectangle
        box = cv2.boxPoints(rect)

        (x, y), (width, height), rect_angle = rect

        """ angle adjustment"""
        angle = 90+rect_angle if width>height else rect_angle

        image = cv2.drawContours(image, [box.astype(np.int0)], -1, (0,255,0), 3) # green

        """ vehicle cropping using perspective transform (classifier input) """
        width, height = int(width), int(height)
        src_pts = box.astype(np.float32)
        dst_pts = np.float32([[0,height], [0,0], [width,0], [width,height]])

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        img_vehicle_crop = cv2.warpPerspective(image, M, (width, height))

        """ crop image -90 degree rotation """
        if width > height:
            img_vehicle_crop = cv2.transpose(img_vehicle_crop)
            img_vehicle_crop = cv2.flip(img_vehicle_crop, flipCode=0)
        # cv2.imshow("crop", img_vehicle_crop)

        """ Classifier input pre-processing """
        img_vehicle_crop = cv2.resize(img_vehicle_crop, (40, 80))
        img_vehicle_crop = cv2.cvtColor(img_vehicle_crop,cv2.COLOR_BGR2GRAY)
        input_x = img_vehicle_crop.flatten()

        """ Classify """
        out = classifier.predict(input_x[None, :])

        """ If output is "down", then angle+=180 """
        if out[0] == 2:
            angle += 180

        """ angle re-arrange (0~360) """
        angle = 360 + angle if angle < 0 else angle
        angle = angle - 360 if angle > 360 else angle

        image = cv2.putText(image, "heading : %d" % (angle), tuple(box[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), thickness=2)

    return image


def Space_Detector(image, space):
    _image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    empty_sace_ids = []
    
    for idx, s in enumerate(space):
        (x1, y1), (x2, y2) = s
        space_crop = _image[y1:y2, x1:x2]  # space cropping
        space_crop = cv2.resize(space_crop, (40, 80))  # resize to classifier input size

        """ Classify """
        input_x = space_crop.flatten()

        out = classifier.predict(input_x[None, :])

        if out[0] == 0:  # 0 means empty , 1 and 2 mean occupy
            empty_space_ids.append(idx)
        
    return empty_space_ids
