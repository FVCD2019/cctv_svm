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

    def imageCB(self, data):
        try:
            ipm_image0 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        """ pre-defined space location """
        upper_space = [[(100 + 400 * i, 100), (500 + 400 * i, 900)] for i in range(6)]
        lower_space = [[(100 + 400 * i, 2200), (500 + 400 * i, 3000)] for i in range(4)]

        num_images = 50
        max_vehicle = 4

        for compose in range(num_images):

            print("[%03d / %03d]" % (compose, num_images), end='\r')

            img_space = space_generator(max_vehicle, upper_space, lower_space)

            cv2.imshow("input", cv2.resize(img_space, (600, 600)))

            """ vehicle detect """
            t0 = time.time()
            img_detected = Space_Detector(img_space, upper_space, up=True)
            img_detected = Space_Detector(img_detected, lower_space, up=False)
            t1 = time.time()

            img_detected = cv2.resize(img_detected, (600, 600))
            img_detected = cv2.putText(img_detected, "FPS : %d" % (1 / (t1 - t0)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                       1,
                                       (0, 0, 0), thickness=3)

            cv2.imshow("out", img_detected)
            if compose > 0:
                video_writer.write(img_detected)

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break



""" video writer """
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter('space_output.avi', fourcc, 1.0, (600, 600))

cv2.destroyAllWindows()
video_writer.release()

print("\ndone")

