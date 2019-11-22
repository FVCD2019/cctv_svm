#!/usr/bin/env python
import rospy
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError
from std_msgs.msg import Float32MultiArray

img_size = (1920, 1080)
w, h = 361, 721
k = np.pi/180
trans = np.array([[650], [600]])
crop = np.array([[300],[600]])

class SVM:
    def __init__(self):
        print("init")
        # ROS init
        rospy.init_node('svm', anonymous=True)
        rospy.Subscriber("/ipm0", Image, self.ipmCB0)
        rospy.Subscriber("/ipm1", Image, self.ipmCB1)
        rospy.Subscriber("detector/pose",Float32MultiArray, self.pose) #Ego vehicle pose subscriber
        
        self.image_pub0 = rospy.Publisher("svm0",Image,queue_size=1)
        self.bridge = CvBridge()

    def ipmCB0(self, data):
        try:
            self.cv_image0 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def ipmCB1(self, data):
        try:
            self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def pose(self, data):  #detector pose need to be modified
        self.ego_x = data[0]
        self.ego_y = data[1]
        self.theta = data[2]

        self.car_center = np.array([[self.ego_x], [self.ego_y]])
        self.Rotation = np.array([[math.cos((360 - self.theta)*k), -math.sin((360 - self.theta)*k)],[math.sin((360 - self.theta)*k), math.cos((360 - self.theta)*k)]])

        self.cen_trans = self.car_center - trans
        self.cen_rot = np.round((np.matmul(Rotation, cen_trans)), 0) 
        self.cen_rotated = self.cen_rot + 2*trans

        self.area1 = (int(self.cen_rotated[0]-crop[0]/2), int(self.cen_rotated[1]-crop[1]/2), int(crop[0]/2), int(crop[1])) 
        self.area2 = (int(self.cen_rotated[0]), int(self.cen_rotated[1]-crop[1]/2), int(crop[0]/2), int(crop[1]))

    def stitching(self): #pending
        while not rospy.is_shutdown():
            t1 = time.time()

            # stitching

            self.rows, self.cols = self.cv_image0.shape[:2]
            self.M1 = np.float32([[1,0,650],[0,1,600]]) # translation x+650, y+600
            self.img_trans1 = cv2.warpAffine(self.cv_image0, self.M1, (2*self.cols, 2*self.rows)) 
            self.M1 = cv2.getRotationMatrix2D((1300, 1200), self.theta-360, 1) # rotation center is x=1300, y=1200
            self.dst1 = cv2.warpAffine(self.img_trans1, self.M1, (2*self.cols, 2*self.rows))

            self.dst1_copy = self.dst1.copy()
            self.dst1_copy = self.dst1[self.area1[1]:self.area1[1]+self.area1[3], self.area1[0]:self.area1[0]+self.area1[2]] 



            self.M2 = np.float32([[1,0,650],[0,1,600]]) # translation x+650, y+600
            self.img_trans2 = cv2.warpAffine(self.cv_image1, self.M2, (2*self.cols, 2*self.rows)) 

            self.M2 = cv2.getRotationMatrix2D((1300, 1200), self.theta-360, 1) # rotation center is x=1300, y=1200
            self.dst2 = cv2.warpAffine(self.img_trans2, self.M2, (2*self.cols, 2*self.rows))

            self.dst2_copy = self.dst2.copy()
            self.dst2_copy = self.dst2[self.area2[1]:self.area2[1]+self.area2[3], self.area2[0]:self.area2[0]+self.area2[2]] 

            self.final = cv2.hconcat([self.dst1_copy, self.dst2_copy]) 

            self.final[160:440, 60:240, :] = 0

            t3 = time.time()

            print("FPS : ", 1 / (t3-t1))

            # show the frame and update the FPS counter
            self.image_pub0.publish(self.bridge.cv2_to_imgmsg(self.final,"bgr8"))

            #image_p
            #if cv2.waitKey(1) & 0xFF == ord('q'):
                #    break

    def run(self):
        self.stitching()

###########main#
svm = SVM()
time.sleep(1)
svm.run()
