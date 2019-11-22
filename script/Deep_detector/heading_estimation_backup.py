import rospy
import cv2
import glob
import numpy as np

from detector import Detector
from utils import get_ipm_matrix, perspective_transform

ipm_matrix = get_ipm_matrix()

Det = Detector()

cap = cv2.VideoCapture(0)
cap.set(3, 1920)  # width
cap.set(4, 1080)  # height

#data_dir = '/home/siit/Desktop/Vehicle/DB/single_camera/'
#img_paths = glob.glob(data_dir+"*.jpg")

while True:
    ret, img = cap.read()

    #img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = perspective_transform(img, ipm_matrix)

    h, w, _ = img.shape

    output = Det.forward(img)
    results = Det.post_processing(output)

    img = cv2.resize(img, (512, 512))

    for result in results:
        center = result['center']
        box = result['rbox']
        heading = result['heading']

        img = cv2.drawContours(img, [box.astype(np.int0)], -1, (0, 255, 0), 3)  # green
        img = cv2.putText(img, "(%d,%d) / %d" % (center[0], center[1], heading),
                          tuple(box[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                           (0, 0, 0), thickness=2)

    img = cv2.resize(img, (960, 960))

    cv2.imshow("output", img[:, :, ::-1])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
