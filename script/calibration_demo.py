#!/usr/bin/env python
import rospy
import numpy as np
import cv2
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

img_size = (1920, 1080)

K = np.array(
    [[692.664263, 0, 991.0584881],
     [0, 693.59339992, 606.92591469],
     [0, 0, 1]] ,dtype=np.float32)

D = np.array([0.01199269, -0.01029937, 0, 0, 0])

class DISTORT:
    def __init__(self):
        print("init")
        # Ros init
        rospy.init_node('cam_cali', anonymous=True)
        rospy.Subscriber("/cam0", Image, self.imgCB0)
        self.image_pub0 = rospy.Publisher("distort_cam0", Image, queue_size=1)
        self.bridge = CvBridge()


    def imgCB0(self, data):
        try:
            self.cv_image0 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)


    def undistortion(self, cam_id0=1):
        print("undistort")
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, D, img_size, 1, img_size)
        mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, newcameramtx, img_size, 5)
	r = rospy.Rate(30)
        # loop over frames from the video file stream
        while not rospy.is_shutdown():
            t1 = time.time()

            # undistort
            dst0 = cv2.remap(self.cv_image0, mapx, mapy, cv2.INTER_LINEAR)

            t3 = time.time()

            print("FPS : ", 1 / (t3-t1))

            # show the frame and update the FPS counter
            self.image_pub0.publish(self.bridge.cv2_to_imgmsg(dst0,"bgr8"))
            #image_pub2.publish(bridge.cv2_to_imgmsg(dst2, "bgr8"))
            #cv2.imshow("Frame", dst)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
	    r.sleep()

    def run(self):
        #self.undistortion(cam_id0=0)
        self.undistortion(cam_id0=1)


######### MAIN ###########
distort = DISTORT()
time.sleep(1)
distort.run()
