#!/usr/bin/env python
import rospy
import cv2
import time
import numpy as np
from std_msgs.msg import String
from std_msgs.msg import Int16MultiArray
from generator import space_generator
from detector import Space_Detector
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError

class PSPACE:
    def __init__(self):
        print("init")
        rospy.init_node('p_space', anonymous=True)
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher("p_space_image",Image,queue_size=1)
        self.pspace_pub = rospy.Publisher("p_space_id",Int16MultiArray ,queue_size=1)
        self.image_sub = rospy.Subscriber("/stitch", Image, self.imageCB)
        self.pspace_info = Int16MultiArray()
        self.pspace_info.data = []

    def imageCB(self, data):
        try:
            ipm_image0 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        """ pre-defined space location """
        upper_space = [[(100 + 400 * i, 100), (500 + 400 * i, 900)] for i in range(6)]
        lower_space = [[(100 + 400 * i, 2200), (500 + 400 * i, 3000)] for i in range(4)]

        pre_defined_space = upper_space + lower_space # list index is space_id
        
        #num_images = 50
        #max_vehicle = 4
        # for compose in range(num_images):
        #
        #     print("[%03d / %03d]" % (compose, num_images), end='\r')
        #
        #     img_space = space_generator(max_vehicle, upper_space, lower_space)
        #
        #     cv2.imshow("input", cv2.resize(img_space, (600, 600)))

        """ empty space recognition """
        empty_space_ids = Space_Detector(ipm_image0, pre_defined_space)
        
        # this is for local ryu
        self.pspace_info.data = []

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(p_detected_img,"bgr8"))
        self.pspace_pub.publish(args, kwds)
        


""" video writer """
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter('space_output.avi', fourcc, 1.0, (600, 600))

cv2.destroyAllWindows()
video_writer.release()

print("\ndone")
