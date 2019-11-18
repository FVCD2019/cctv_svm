#!/usr/bin/env python
import rospy
import numpy as np
import cv2
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError

img_size = (1920, 1080)
width, height = 1300, 1200
# src_pts = np.float32(
#     [[(681, 557), (1177, 597), (1268 , 808), (503, 784)],
#     [(668, 494), (1203, 560), (1292 , 786), (563, 733)],
#     [(671, 510), (1229, 493), (1301 , 721), (587, 750)],
#     [(751, 428), (1276, 498), (1377, 773), (653, 616)]]
# )
# dst_pts = np.float32([(0, 0), (width, 0), (width, height), (0, height)] )
# IPM_matrix = [cv2.getPerspectiveTransform(src_pts[n-1], dst_pts) for n in range(1, 5)]

class IPM:
    IPM_matrix = []
    def __init__(self):
        print("init")
        rospy.init_node('ipm', anonymous=True)
        self.bridge = CvBridge()
        self.image_pub0 = rospy.Publisher("ipm0",Image,queue_size=1)
        self.image_pub1 = rospy.Publisher("ipm1",Image,queue_size=1)
        self.image_sub0 = rospy.Subscriber("/distort_cam0", Image, self.ipmCB0)
        self.image_sub1 = rospy.Subscriber("/distort_cam1", Image, self.ipmCB1)

        self.src_pts0 = np.float32([[753, 170], [747, 351], [735, 649], [730, 841], [1270, 192], [1273, 364], [1065, 656], [1061, 845]])
	self.src_pts1 = np.float32([[1805, 965], [1673, 656], [1522, 344], [1458, 221], [805, 954], [859, 581], [1174, 288], [1158, 162]])
        self.dst_pts = np.float32([[111, 5], [111, 315], [111, 815], [111, 1125], [1011, 5], [1011, 315], [651, 815], [651, 1125]])
        
	#self.IPM_matrix = [cv2.getPerspectiveTransform(self.src_pts[n-1], self.dst_pts) for n in range(1, 5)]
	self.IPM_matrix_0, mask = cv2.findHomography(self.src_pts0, self.dst_pts)
	self.IPM_matrix_1, mask = cv2.findHomography(self.src_pts1, self.dst_pts)


    def ipmCB0(self, data):
        try:
            cv_image0 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        ##cv2.imshow("img0", cv_image0)
        # IPM
        img_IPM0 = cv2.warpPerspective(cv_image0, self.IPM_matrix_0, (width, height))
        ##cv2.imshow("img_IPM0", img_IPM0)
        self.image_pub0.publish(self.bridge.cv2_to_imgmsg(img_IPM0,"bgr8"))
        # show the frame and update the FPS counter
        ##cv2.waitKey(3)


    def ipmCB1(self, data):
        try:
            cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        ##cv2.imshow("img0", cv_image1)
        # IPM
        img_IPM1 = cv2.warpPerspective(cv_image1, self.IPM_matrix_1, (width, height))
        ##cv2.imshow("img_IPM1", img_IPM1)
        self.image_pub1.publish(self.bridge.cv2_to_imgmsg(img_IPM1,"bgr8"))
        # show the frame and update the FPS counter
        ##cv2.waitKey(3)

    def run(self):
        print("run")
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down")
        cv2.destroyAllWindows()


###########main#
ipm = IPM()
time.sleep(1)
ipm.run()
