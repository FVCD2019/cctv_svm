import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import glob

img_size = (1920, 1080)

K = np.array(
[[692.664263, 0, 991.0584881],
[0, 693.59339992, 606.92591469],
[0, 0, 1]] ,dtype=np.float32)

D = np.array([0.01199269, -0.01029937, 0, 0, 0])

#newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, D, img_size, 1, img_size)
#mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, newcameramtx, img_size, 5)

src_pts1 = np.float32([[753, 170], [747, 351], [735, 649], [730, 841], [1270, 192], [1273, 364], [1065, 656], [1061, 845]])
src_pts2 =  np.float32([[1805, 965], [1673, 656], [1522, 344], [1458, 221], [805, 954], [859, 581], [1174, 288], [1158, 162]]) # 쌉고정
dst_pts = np.float32([[111, 5], [111, 315], [111, 815], [111, 1125], [1011, 5], [1011, 315], [651, 815], [651, 1125]])

IPM_matrix1, mask = cv2.findHomography(src_pts1, dst_pts)
IPM_matrix2, mask = cv2.findHomography(src_pts2, dst_pts)


# IPM
img_undistort1 = cv2.imread("C:\\Users\\user\\Desktop\\undis_cam1.jpg")
img_undistort2 = cv2.imread("C:\\Users\\user\\Desktop\\undis_cam2.jpg")
width, height = 1300, 1200

img_IPM1 = cv2.warpPerspective(img_undistort1, IPM_matrix1, (width, height))
img_IPM2 = cv2.warpPerspective(img_undistort2, IPM_matrix2, (width, height))

cv2.imshow("img_IPM", cv2.resize(img_IPM1, (1300, 1200)))
cv2.imshow("img_IPM2", cv2.resize(img_IPM2, (1300, 1200)))
#cv2.imwrite("C:\\Users\\user\\Desktop\\IPM_cam2.jpg", img_IPM)
cv2.waitKey()
