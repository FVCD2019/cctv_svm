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
    #def undistortion(self,cam_id0=0, cam_id1=1, cam_id=2):
    def undistortion(self, cam_id0=0):
        print("undistort")
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, D, img_size, 1, img_size)
        mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, newcameramtx, img_size, 5)
        resource0 = "/dev/video" + str(cam_id0)
        resource1 = "/dev/video" + str(cam_id1)

        cap0 = cv2.VideoCapture(resource0)
        cap1 = cv2.VideoCapture(resource1)
        

        if (cap0.isOpened() and cap1.isOpened()):
            image_pub0 = rospy.Publisher("distort_cam0",Image, queue_size=1)
            image_pub1 = rospy.Publisher("distort_cam1",Image, queue_size=1)
            bridge = CvBridge()

            # loop over frames from the video file stream
            while not rospy.is_shutdown():
                t1 = time.time()
                rval0, frame0 = cap0.read()
                rval1, frame1 = cap1.read()
                frame0 = cv2.resize(frame0, img_size)
                frame1 = cv2.resize(frame1, img_size)
               

                # undistort
                dst0 = cv2.remap(frame0, mapx, mapy, cv2.INTER_LINEAR)
                dst1 = cv2.remap(frame1, mapx, mapy, cv2.INTER_LINEAR)
                
                '''# crop the image
                x, y, w, h = roi
                dst0 = dst0[y:y + h, x:x + w]
                #dst1 = dst1[y:y + h, x:x + w]
                #dst2 = dst2[y:y + h, x:x + w]'''
                t3 = time.time()

                print("FPS : ", 1 / (t3-t1))

                # show the frame and update the FPS counter
                #dst0 = cv2.resize(dst0, (1280, 720))
                #dst1 = cv2.resize(dst1, (1280, 720))
                #dst2 = cv2.resize(dst2, (1280, 720))
                image_pub0.publish(bridge.cv2_to_imgmsg(dst0,"bgr8"))
                image_pub1.publish(bridge.cv2_to_imgmsg(dst1, "bgr8"))
                #image_pub2.publish(bridge.cv2_to_imgmsg(dst2, "bgr8"))
                #cv2.imshow("Frame", dst)
                #if cv2.waitKey(1) & 0xFF == ord('q'):
                #    break
        else :
            print("unable to open cam")

    def run(self):
        print("init")
        # Ros init
        rospy.init_node('cam_cali', anonymous=True)
        #self.undistortion(cam_id0=0)
        self.undistortion(cam_id0=0,cam_id1=1)


######### MAIN ###########
distort = DISTORT()
distort.run()
