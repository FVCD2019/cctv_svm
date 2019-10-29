#!/usr/bin/env python
import rospy
import numpy as np
import glob
import cv2
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError

img_size = (1920, 1080)
width, height = 1920, 1080

class STITCHING:
    def __init__(self):
        print("init")
        rospy.init_node('stitiching', anonymous=True)
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher("stitch",Image,queue_size=1)
        self.image_sub0 = rospy.Subscriber("/ipm0", Image, self.imageCB0)
        self.image_sub1 = rospy.Subscriber("/ipm1", Image, self.imageCB1)
        self.cb1 = False
        self.cb2 = False

    def imageCB0(self, data):
        try:
            self.ipm_image0 = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.cb1 = True
        except CvBridgeError as e:
            print(e)
            self.cb1 = False

    def imageCB1(self, data):
        try:
            self.ipm_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.cb2 = True
        except CvBridgeError as e:
            print(e)
            self.cb2 = False

    def dostitching(self):
        while not rospy.is_shutdown():
            if self.cb1 == True and self.cb2 == True:
                self.image_stitch = cv2.addWeighted(self.ipm_image0, 0.5, self.ipm_image1, 0.5, 0)
                cv2.resize(self.image_stitch, (width, height))
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.image_stitch,"bgr8"))
            else :
                print("cb not executed")

########MAIN#######
stitch = STITCHING()
stitch.dostitching()