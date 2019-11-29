#!/usr/bin/env python
import rospy
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import time
import Queue
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError
from std_msgs.msg import Float32MultiArray


img_size = (1920, 1080)
w, h = 361, 721
k = np.pi/180
trans = np.array([[650], [600]])
crop = np.array([[300],[600]])
crop2 = np.array([[200],[400]])
camera_cen_x = 468
camera_cen_y = 814

class SVM:
    def __init__(self):
        print("init")
        # ROS init
        rospy.init_node('svm', anonymous=True)
	self.bridge = CvBridge()
	self.revised_car_center = []
	self.car_center = []
        rospy.Subscriber("/ipm0", Image, self.ipmCB0)
        rospy.Subscriber("/ipm1", Image, self.ipmCB1)
        rospy.Subscriber("detector/pose",Float32MultiArray, self.pose) #Ego vehicle pose subscriber
        rospy.Subscriber("detector/box",Float32MultiArray, self.rect)

        
        self.image_pub0 = rospy.Publisher("svm0",Image,queue_size=1)
	self.box = []
        self.queue = Queue.Queue()

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


    def image_queue(self, img):
        self.queue.put(img)


    def rect(self, data):
        
	self.box = np.array(data.data)
	self.box = self.box.reshape(4,2)

    def pose(self, data): 
	global camera_cen_x, camera_cen_y 
        self.ego_x = data.data[0]
        self.ego_y = data.data[1]
        self.theta = data.data[2]

        dis = [0,0,0,0]
        min_1 = [9999, 0]

	#dis[0] = math.sqrt( math.pow(camera_cen_x - self.box[0,0], 2) +  math.pow(camera_cen_y - self.box[0,1], 2) )
	#dis[1] = math.sqrt( math.pow(camera_cen_x - self.box[1,0], 2) +  math.pow(camera_cen_y - self.box[1,1], 2) )
	#dis[2] = math.sqrt( math.pow(camera_cen_x - self.box[2,0], 2) +  math.pow(camera_cen_y - self.box[2,1], 2) )
	#dis[3] = math.sqrt( math.pow(camera_cen_x - self.box[3,0], 2) +  math.pow(camera_cen_y - self.box[3,1], 2) )
	
	aa = [math.sqrt( math.pow(camera_cen_x - self.box[0,0], 2) +  math.pow(camera_cen_y - self.box[0,1], 2) )]
	bb = [math.sqrt( math.pow(camera_cen_x - self.box[1,0], 2) +  math.pow(camera_cen_y - self.box[1,1], 2) )]
	cc = [math.sqrt( math.pow(camera_cen_x - self.box[2,0], 2) +  math.pow(camera_cen_y - self.box[2,1], 2) )]
	dd = [math.sqrt( math.pow(camera_cen_x - self.box[3,0], 2) +  math.pow(camera_cen_y - self.box[3,1], 2) )]

	dis = [aa,bb,cc,dd]

        for i in range(4):

            #dis[i] = 0 # math.sqrt((camera_cen_x - self.box[i,0])**2+(camera_cen_y - self.box[i,1])**2)
            #print("camera_cen_x : ", type(camera_cen_x))
            #print("camera_cen_y : ", type(camera_cen_y))
	    #print("box0", type(self.box[i, 0]))
	    #print("box1", type(self.box[i, 1]))

            if (dis[i] < min_1[0]):
                min_1[0] = dis[i]
                min_1[1] = i


        revised_car_center = [0,0]
        Rotation = np.array([[math.cos(k*self.theta), -math.sin(k*self.theta)],[math.sin(k*self.theta), math.cos(k*self.theta)]])

        ori_mat = np.array([[60, 60, -60, -60], [118, -118, 118, -118]])
        o_x = self.box[min_1[1],0]
        o_y = self.box[min_1[1],1]
        c_x = self.ego_x 
        c_y = self.ego_y 
        ori_point = np.array([[o_x,o_x,o_x,o_x],[o_y,o_y,o_y,o_y]])

        new_mat = np.matmul(Rotation, ori_mat)+ori_point

        cen_mat = np.array([[c_x,c_x,c_x,c_x],[c_y,c_y,c_y,c_y]])
        dis_mat = new_mat - cen_mat
        dis_mat = np.matmul(np.transpose(dis_mat), dis_mat)

        min = np.min([dis_mat[0,0],dis_mat[1,1],dis_mat[2,2],dis_mat[3,3]])

        idx = 0 # idx is number of index order about candidate 4 center point
        for j in range(4):
            if(dis_mat[j,j] == min):
                idx = j

        self.revised_car_center=np.array([new_mat[0,idx], new_mat[1,idx]])
	
        ##### for stitching
        self.car_center = np.array([[self.revised_car_center[0]], [self.revised_car_center[1]]])
        self.Rotation = np.array([[math.cos((360 - self.theta)*k), -math.sin((360 - self.theta)*k)],[math.sin((360 - self.theta)*k), math.cos((360 - self.theta)*k)]])

        self.cen_trans = self.car_center - trans 
        self.cen_rot = np.round((np.matmul(Rotation, self.cen_trans)), 0) 
        self.cen_rotated = self.cen_rot + 2*trans

        ## for big cropping
        self.area1 = (int(self.cen_rotated[0]-crop[0]/2), int(self.cen_rotated[1]-crop[1]/2), int(crop[0]/2), int(crop[1])) 
        self.area2 = (int(self.cen_rotated[0]), int(self.cen_rotated[1]-crop[1]/2), int(crop[0]/2), int(crop[1]))

        ## for small cropping
        self.area3 = (int(self.cen_rotated[0]-crop2[0]/2), int(self.cen_rotated[1]-crop2[1]/2), int(crop2[0]/2), int(crop2[1]))
        self.area4 = (int(self.cen_rotated[0]), int(self.cen_rotated[1]-crop2[1]/2), int(crop2[0]/2), int(crop2[1]))

    def stitching(self): #pending
        while not rospy.is_shutdown():

            if self.queue.qsize() < 30:
                continue

            t1 = time.time()

            prev_img1 = self.queue.get() #image before 30frame
            prev_img2 = prev_img1.copy()
            # stitching

            self.rows, self.cols = self.cv_image0.shape[:2]
            self.M1 = np.float32([[1,0,650],[0,1,600]]) # translation x+650, y+600
            self.img_trans1 = cv2.warpAffine(self.cv_image0, self.M1, (2*self.cols, 2*self.rows))            
            self.img_trans3 = cv2.warpAffine(self.prev_image1, self.M1, (2*self.cols, 2*self.rows))


            self.M1 = cv2.getRotationMatrix2D((1300, 1200), self.theta-360, 1) # rotation center is x=1300, y=1200
            self.dst1 = cv2.warpAffine(self.img_trans1, self.M1, (2*self.cols, 2*self.rows))
            self.dst3 = cv2.warpAffine(self.img_trans3, self.M1, (2*self.cols, 2*self.rows))


            self.dst1_copy = self.dst1.copy()
            self.dst3_copy = self.dst3.copy()

            self.dst1_copy = self.dst1[self.area1[1]:self.area1[1]+self.area1[3], self.area1[0]:self.area1[0]+self.area1[2]] 
            self.dst3_copy = self.dst3[self.area3[1]:self.area3[1]+self.area3[3], self.area3[0]:self.area3[0]+self.area3[2]]

            self.M2 = np.float32([[1,0,650],[0,1,600]]) # translation x+650, y+600
            self.img_trans2 = cv2.warpAffine(self.cv_image1, self.M2, (2*self.cols, 2*self.rows)) 
            self.img_trans4 = cv2.warpAffine(self.prev_image2, self.M2, (2*self.cols, 2*self.rows))

            self.M2 = cv2.getRotationMatrix2D((1300, 1200), self.theta-360, 1) # rotation center is x=1300, y=1200
            self.dst2 = cv2.warpAffine(self.img_trans2, self.M2, (2*self.cols, 2*self.rows))
            self.dst4 = cv2.warpAffine(self.img_trans4, self.M2, (2*self.cols, 2*self.rows))

            self.dst2_copy = self.dst2.copy()
            self.dst4_copy = self.dst4.copy()
            self.dst2_copy = self.dst2[self.area2[1]:self.area2[1]+self.area2[3], self.area2[0]:self.area2[0]+self.area2[2]] 
            self.dst4_copy = self.dst4[self.area4[1]:self.area4[1]+self.area4[3], self.area4[0]:self.area4[0]+self.area4[2]]

            self.final = cv2.hconcat([self.dst1_copy, self.dst2_copy]) 
            self.final2 = cv2.hconcat([self.dst3_copy, self.dst4_copy])


            # for inserting small cropping img to big cropping img
            self.final[100:500, 50:250, :] = self.final2[0:400, 0:200, :]
            #self.final[180:420, 90:210, :] = self.topcar[0:240, 0:120, :] # have to load the topcar img (size is 240, 120)
	    self.final[180:420, 90:210, :] = 0
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
