#!/usr/bin/env python
import rospy
import cv2
import time
import numpy as np
#from std_msgs.msg import String
from std_msgs.msg import Int16MultiArray
from std_msgs.msg import Int16
#from generator import space_generator
from detector import Space_Detector
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError

class PSPACE:
    def __init__(self):
        print("init")
        rospy.init_node('p_space', anonymous=True)
        self.bridge = CvBridge()
        self.pspace_pub = rospy.Publisher("detector/p_space",Int16MultiArray ,queue_size=1)
        self.pspace_id_pub = rospy.Publisher("p_space_id",Int16 ,queue_size=1)
        self.image_sub = rospy.Subscriber("/ipm0", Image, self.imageCB)
        self.pspace_info = Int16MultiArray()
        self.pspace_info.data = []
        self.ps_id = 0
        self.ps_x = 0
        self.ps_y = 0

    def imageCB(self, data):
        try:
            self.stitch_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def run(self):
        while not rospy.is_shutdown():
            """ pre-defined space location """
            upper_space = [[(110,0),(294,318)], [(294,0),(470,318)], [(470,0),(651,318)], [(651,0),(830,318)], [(830,0),(1009,318)]]
            lower_space = [[(111,816),(292,1200)], [(292,816),(472,1200)], [(472,816),(651,1200)]]

            pre_defined_space = upper_space + lower_space # list index is space_id

            """ empty space recognition """
            empty_space_ids = Space_Detector(self.stitch_image, pre_defined_space)
            self.ps_id = empty_space_ids[0]
            self.ps_u = pre_defined_space[self.ps_id][0]
            self.ps_b = pre_defined_space[self.ps_id][1]

            mx = (self.ps_u[0] + self.ps_b[0]) / 2
            my = (self.ps_u[0] + self.ps_b[0]) / 2
            # this is for local ryu
            self.pspace_info.data = [self.ps_id+1, mx, my]
            self.pspace_pub.publish(self.pspace_info)
            self.pspace_id_pub.publish(self.ps_id+1)

###########MAIN############
pspace = PSPACE()
time.sleep(1)
pspace.run()
