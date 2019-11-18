import numpy as np
import cv2

img_size = (1920, 1080)

K = np.array(
[[692.664263, 0, 991.0584881],
[0, 693.59339992, 606.92591469],
[0, 0, 1]] ,dtype=np.float32)

D = np.array([0.01199269, -0.01029937, 0, 0, 0])

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, D, img_size, 1, img_size)
mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, newcameramtx, img_size, 5)

images = cv2.imread("C:\\Users\\user\\Desktop\\cam1.jpg")

img_undistort = cv2.remap(images, mapx, mapy, cv2.INTER_LINEAR)

cv2.imshow("img_undistort", cv2.resize(img_undistort, (1920, 1080)))
cv2.imwrite("C:\\Users\\user\\Desktop\\undis_cam1.jpg", img_undistort)
cv2.waitKey()
