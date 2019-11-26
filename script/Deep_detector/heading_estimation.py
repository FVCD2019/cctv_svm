#!/usr/bin/env python
import rospy
import cv2
import numpy as np
import time

from detector import Detector
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Int16MultiArray
from std_msgs.msg import Int16
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError


class CARPOSE:
    def __init__(self):
        print("init")
        rospy.init_node('detector', anonymous=True)
        self.bridge = CvBridge()
        self.pose_pub = rospy.Publisher("detector/pose",Float32MultiArray ,queue_size=1)
        self.pspace_pub = rospy.Publisher("detector/p_space",Int16MultiArray ,queue_size=1)
        self.pspace_id_pub = rospy.Publisher("p_space_id",Int16 ,queue_size=1)
        self.image_sub = rospy.Subscriber("/ipm0", Image, self.imageCB)
        self.pspace_info = Int16MultiArray()
        self.pspace_info.data = []
        self.pose_info = Float32MultiArray()
        self.pose_info.data = []
        self.ego_x = 0.0
        self.ego_y = 0.0
        self.ego_heading = 0.0
        self.Det = Detector()
        print("init end")

    def imageCB(self, data):
        try:
            self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def run(self):
        FPS = 0
        count = 0

        while not rospy.is_shutdown():
            count += 1

            start = time.time()

            img = self.image.copy()
            img = img[:, :, ::-1]

            h, w, _ = img.shape
            
            output = self.Det.forward(img)

            results = self.Det.post_processing(output)

            img, empty_space = self.Det.space_recognition(img)

            (Pid, center_x, center_y) = empty_space
            center_x = center_x
            center_y = center_y

            self.pspace_info.data = [Pid, center_x, center_y]
	    #self.pspace_info.data = [Pid, center_y, center_x]
            self.pspace_pub.publish(self.pspace_info)
            self.pspace_id_pub.publish(Pid)

            end = time.time()
            FPS += 1 / (end - start)

            for result in results:
                center = result['center']
                box = result['rbox']
                heading = result['heading']

                center_rescale = np.float32([w/512., h/512.])
                box_rescale = np.tile(center_rescale, (4, 1))

                center *= center_rescale
                box *= box_rescale

                center[0] = center[0]
                center[1] = 1200 - center[1]
		self.pose_info.data = [center[0], center[1], heading]
		#self.pose_info.data = [center[1], center[0], heading]
		self.pose_pub.publish(self.pose_info)

                #img = cv2.drawContours(img.astype(np.uint8), [box.astype(np.int0)], -1, (0, 0, 255), 3)  # green
                #img = cv2.putText(img, "(%d,%d) / %d" % (center[0], center[1], heading),
                #                tuple(box[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                #                (0, 0, 0), thickness=2)
            '''
            img = cv2.putText(img, "FPS : %.2f" % (FPS / count), (w-200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                       (0, 0, 0), thickness=2)
                
            output = output[0, :, :, 0].cpu().detach().numpy()
            output = np.clip(output * 255, 0, 255)
            output = cv2.applyColorMap(output.astype(np.uint8), cv2.COLORMAP_JET)

            output = cv2.resize(output, (512, 512))
            img = cv2.resize(img, (512, 512))

            result_img = cv2.hconcat([img[:, :, ::-1], output])
            cv2.imshow("output", result_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        '''

#######MAIN######
carpose = CARPOSE()
time.sleep(1)
carpose.run()
