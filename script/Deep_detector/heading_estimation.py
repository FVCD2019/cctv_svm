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
        print("init end")

    def imageCB(self, data):
        try:
            self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def run(self):
        while not rospy.is_shutdown():
            img = self.image.copy()
            img = img[:, :, ::-1]

            h, w, _ = img.shape

            output = self.Det.forward(img)
            results = self.Det.post_processing(output)

            for result in results:
                center = result['center']
                box = result['rbox']
                heading = result['heading']

                center_rescale = np.float32([w/512., h/512.])
                box_rescale = np.tile(center_rescale, (4, 1))

                center *= center_rescale
                box *= box_rescale

                img = cv2.drawContours(img.astype(np.uint8), [box.astype(np.int0)], -1, (0, 255, 0), 3)  # green
                img = cv2.putText(img, "(%d,%d) / %d" % (center[0], center[1], heading),
                                tuple(box[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
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


#######MAIN######
carpose = CARPOSE()
time.sleep(1)
carpose.run()
