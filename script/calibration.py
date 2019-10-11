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
    def undistortion(self,cam_id=0):
        print("undistort")
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, D, img_size, 1, img_size)
        mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, newcameramtx, img_size, 5)
        resource = "/dev/video" + str(cam_id)
        cap = cv2.VideoCapture(resource)

        if (cap.isOpened()):
            image_pub0 = rospy.Publisher("distort_cam0",Image, queue_size=1)
            #image_pub1 = rospy.Publisher("distort_cam1",Image, queue_size=1)
            #image_pub2 = rospy.Publisher("distort_cam2",Image, queue_size=1)
            #image_pub3 = rospy.Publisher("distort_cam3",Image, queue_size=1)
            bridge = CvBridge()

            # loop over frames from the video file stream
            while not rospy.is_shutdown():
                t1 = time.time()
                rval, frame = cap.read()
                frame = cv2.resize(frame, img_size)

                #cv2.imshow("original", cv2.resize(frame, (1280, 720)))

                # undistort
                dst = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

                # crop the image
                x, y, w, h = roi
                dst = dst[y:y + h, x:x + w]
                t3 = time.time()

                print("FPS : ", 1 / (t3-t1))

                # show the frame and update the FPS counter
                dst = cv2.resize(dst, (1280, 720))
                image_pub0.publish(bridge.cv2_to_imgmsg(dst,"bgr8"))
                #cv2.imshow("Frame", dst)
                #if cv2.waitKey(1) & 0xFF == ord('q'):
                #    break
        else :
            print("unable to open cam")

    def run(self):
        print("init")
        # Ros init
        rospy.init_node('cam_cali', anonymous=True)
        self.undistortion(cam_id=0)


######### MAIN ###########
distort = DISTORT()
distort.run()
