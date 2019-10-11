import numpy as np
import cv2
import time

img_size = (1920, 1080)

K = np.array(
    [[692.664263, 0, 991.0584881],
     [0, 693.59339992, 606.92591469],
     [0, 0, 1]] ,dtype=np.float32)

D = np.array([0.01199269, -0.01029937, 0, 0, 0])

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, D, img_size, 1, img_size)
mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, newcameramtx, img_size, 5)

###############################################################################################

width, height = 1920, 1080
img_list = "/home/siit/Pictures/Webcam/top%d.jpg"

src_pts = np.float32(
    [[(681, 557), (1177, 597), (1268 , 808), (503, 784)],
    [(668, 494), (1203, 560), (1292 , 786), (563, 733)],
    [(671, 510), (1229, 493), (1301 , 721), (587, 750)],
    [(751, 428), (1276, 498), (1377, 773), (653, 616)]]
)
dst_pts = np.float32([(0, 0), (width, 0), (width, height), (0, height)] )

IPM_matrix = [cv2.getPerspectiveTransform(src_pts[n-1], dst_pts) for n in range(1, 5)]

for n in range(1, 5):
    img = cv2.imread(img_list % n)
    t1 = time.time()

    img= cv2.resize(img, img_size)

    t2 = time.time()
    print("Resize : ", 1/(t2-t1))

    cv2.imshow("img", cv2.resize(img, (1280, 720)))

    t3 = time.time()

    # undistort
    img_undistort = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    # crop the image
    x, y, w, h = roi
    img_undistort = img_undistort[y:y + h, x:x + w]
    t4 = time.time()
    print("undistort : ", 1 / (t4 - t3))

    cv2.imshow("img_undistort", cv2.resize(img_undistort, (1280, 720)))

    # IPM
    t5 = time.time()
    img_IPM = cv2.warpPerspective(img_undistort, IPM_matrix[n-1], (width, height))
    t6 = time.time()
    print("IPM : ", 1 / (t6 - t5))

    cv2.imshow("img_IPM", cv2.resize(img_IPM, (1280, 720)))

    print("FPS : ", 1 / (t6-t5+t4-t3+t2-t1))
    print()

    # show the frame and update the FPS counter

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
