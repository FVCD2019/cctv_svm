#!/usr/bin/env python
import rospy
import cv2
import time
import numpy as np
#from std_msgs.msg import String
from std_msgs.msg import Int16MultiArray
#from generator import space_generator
from detector import Space_Detector
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError

class PSPACE:
    def __init__(self):
        print("init")
        rospy.init_node('p_space', anonymous=True)
        self.bridge = CvBridge()
        self.pspace_pub = rospy.Publisher("p_space_id",Int16MultiArray ,queue_size=1)
        self.image_sub = rospy.Subscriber("/stitch", Image, self.imageCB)
        self.pspace_info = Int16MultiArray()
        self.pspace_info.data = []
        self.ps_id = 0
        self.ps_x = 0
        self.ps_y = 0

    def imageCB(self, data):
        print("cb")
        try:
            stitch_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            print("cb1")
        except CvBridgeError as e:
            print(e)
            print("cb2")

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
        empty_space_ids = Space_Detector(stitch_image, pre_defined_space)
        self.ps_id = empty_space_ids+1
        self.ps_x = pre_defined_space[empty_space_ids][0]
        self.ps_y = pre_defined_space[empty_space_ids][1]

        # this is for local ryu
        self.pspace_info.data = [self.ps_id, self.ps_x, self.ps_y]
        print("pub")
        #self.pspace_pub.publish(self.pspace_info)

###########MAIN############
pspace = PSPACE()
time.sleep(1)
