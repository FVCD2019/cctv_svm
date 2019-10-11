import numpy as np
import cv2
import time

from imutils.video import FileVideoStream

img_size = (1920, 1080)

K = np.array(
    [[692.664263, 0, 991.0584881],
     [0, 693.59339992, 606.92591469],
     [0, 0, 1]] ,dtype=np.float32)

D = np.array([0.01199269, -0.01029937, 0, 0, 0])

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, D, img_size, 1, img_size)
mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, newcameramtx, img_size, 5)

fvs = FileVideoStream(1).start()

# loop over frames from the video file stream
while fvs.more():
    t1 = time.time()

    frame = fvs.read()
    frame = cv2.resize(frame, img_size)

    t2 = time.time()

    cv2.imshow("original", cv2.resize(frame, (1280, 720)))

    t3 = time.time()

    # undistort
    dst = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    t4 = time.time()

    print("FPS : ", 1 / (t4-t3+t2-t1))

    # show the frame and update the FPS counter
    dst = cv2.resize(dst, (1280, 720))

    cv2.imshow("Frame", dst)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
