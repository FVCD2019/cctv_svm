#!/usr/bin/env python
import rospy
import cv2
import glob
import numpy as np
import time

from detector import Detector
from utils import get_ipm_matrix, perspective_transform
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError


class CARPOSE:
    def __init__(self):
        print("init")
        rospy.init_node('car_detect', anonymous=True)
        self.bridge = CvBridge()
        self.pose_pub = rospy.Publisher("detector/pose",Float32MultiArray ,queue_size=1)
        self.image_sub = rospy.Subscriber("/ipm0", Image, self.imageCB)
        self.pose_info = Float32MultiArray()
        self.pose_info.data = []
        self.ego_x = 0.0
        self.ego_y = 0.0
        self.ego_heading = 0.0
        self.Det = Detector()
        self.camera_cen_x = 468
        self.camera_cen_y = 814
        self.a = np.pi/180

    def imageCB(self, data):
        try:
            self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def run(self):
        while not rospy.is_shutdown:
            h, w, _ = self.image.shape 
            output = self.Det.forward(self.image)
            results = self.Det.post_processing(output)

            self.image = cv2.resize(self.image, (512, 512))

************* 리사이즈해서 좌표값 맞춰야됨 **************

            for result in results:
                center = result['center']
                box = result['rbox']
                heading = result['heading']

            dis = [0,0,0,0]
            min_1 = [9999, 0]

            for i in range(4):
                dis[i] = math.sqrt((self.camera_cen_x - box[i,0])**2+(self.camera_cen_y - box[i,1])**2)
                if (dis[i] < min_1[0]):
                    min_1[0] = dis[i]
                    min_1[1] = i


            revised_car_center = [0,0]
            Rotation = np.array([[math.cos(a*heading), -math.sin(a*heading)],[math.sin(a*heading), math.cos(a*heading)]])

            ori_mat = np.array([[60, 60, -60, -60], [118, -118, 118, -118]])
            o_x = box[min_1[1],0]
            o_y = box[min_1[1],1]
            c_x = center[0]
            c_y = center[1]
            ori_point = np.array([[o_x,o_x,o_x,o_x],[o_y,o_y,o_y,o_y]])

            new_mat = np.matmul(Rotation,ori_mat)+ori_point

            cen_mat = np.array([[c_x,c_x,c_x,c_x],[c_y,c_y,c_y,c_y]])
            dis_mat = new_mat - cen_mat
            dis_mat = np.matmul(np.transpose(dis_mat), dis_mat)

            min = np.min([dis_mat[0,0],dis_mat[1,1],dis_mat[2,2],dis_mat[3,3]])

            idx = 0 # idx is number of index order about candidate 4 center point
            for i in range(4):
                if(dis_mat[i,i] == min):
                    idx = i

            revised_car_center=np.array([new_mat[0,idx], new_mat[1,idx])
****************************** pose publish 작성 필요****************************

            self.image = cv2.drawContours(self.image, [box.astype(np.int0)], -1, (0, 255, 0), 3)  # green
            self.image = cv2.putText(self.image, "(%d,%d) / %d" % (center[0], center[1], heading),
                            tuple(box[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 0), thickness=2)

            self.image = cv2.resize(img, (960, 960))

            cv2.imshow("output", self.image[:, :, ::-1])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


#######MAIN######
carpose = CARPOSE()
time.sleep(1)
carpose.run()
