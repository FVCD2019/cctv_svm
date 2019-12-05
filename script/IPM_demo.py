#!/usr/bin/env python
import rospy
import numpy as np
import cv2
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError

img_size = (1920, 1080)
width, height = 1300, 1200

class IPM:
    def __init__(self):
        print("init")
        rospy.init_node('ipm', anonymous=True)
        self.bridge = CvBridge()
        self.image_pub0 = rospy.Publisher("ipm0",Image,queue_size=1)
        self.image_sub0 = rospy.Subscriber("/distort_cam0", Image, self.ipmCB0)

        self.src_pts0 = np.float32([[751, 158], [746, 340], [736, 641], [732, 830], [1272, 180], [1274, 353], [1065, 646], [1061, 834]])
        self.dst_pts = np.float32([[111, 5], [111, 315], [111, 815], [111, 1125], [1011, 5], [1011, 315], [651, 815], [651, 1125]])
        
	#self.IPM_matrix = [cv2.getPerspectiveTransform(self.src_pts[n-1], self.dst_pts) for n in range(1, 5)]
	self.IPM_matrix_0, mask = cv2.findHomography(self.src_pts0, self.dst_pts)


    def ipmCB0(self, data):
        try:
            self.cv_image0 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)


    def run(self):
	r = rospy.Rate(30)
        while not rospy.is_shutdown():
            img_IPM0_cuda = cv2.UMat(self.cv_image0)

	    img_IPM0 = cv2.warpPerspective(img_IPM0_cuda, self.IPM_matrix_0, (width, height))

            img_IPM0 = img_IPM0.get()

	    self.image_pub0.publish(self.bridge.cv2_to_imgmsg(img_IPM0,"bgr8"))
	    r.sleep()



###########main#
ipm = IPM()
time.sleep(2)
ipm.run()
