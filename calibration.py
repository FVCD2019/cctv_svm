'''
img = cv2.imread("C:/Users/user/Desktop/tile3.jpg")
cv2.waitKey(0)
point1 = np.float32([(286,1033), (564, 542), (1518, 1031), (1293,519)])
cv2.circle(img,  (286,1033), 10, (255, 0, 0), -1)
cv2.circle(img, (564, 542), 10, (0, 255, 0), -1)
cv2.circle(img, (1518, 1031), 10, (0, 0, 255), -1)
cv2.circle(img, (1293,519), 10, (255, 255, 255), -1)
print(img.shape)
point2 = np.float32([(0, 600), (0, 0), (1000, 600), (1000, 0)])

mat = cv2.getPerspectiveTransform(point1, point2)
newimg = cv2.warpPerspective(img, mat, (1920,1080))

#cv2.imshow('img', img)
#cv2.imshow('newimg', newimg)
#cv2.waitKey(0)
plt.imshow(img)
plt.show()
plt.imshow(newimg)
plt.show()
'''
import cv2
import numpy as np
import os
import glob
import time

from threading import Thread
from queue import Queue

import imutils

mat_path = "%s"

if not os.path.exists(mat_path % ("img_points.npy")) or \
        not os.path.exists(mat_path % ("obj_points.npy")) or \
        not os.path.exists(mat_path % ("DIM.npy")) or \
        not os.path.exists(mat_path % ("K.npy")) or \
        not os.path.exists(mat_path % ("D.npy")):

    CHECKERBOARD = (9,8)
    subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    _img_shape = None

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob('/home/siit/Pictures/Webcam/*.jpg')

    for fname in images:
        img = cv2.imread(fname)
        if _img_shape == None:
            _img_shape = img.shape[:2]
        else:
            assert _img_shape == img.shape[:2], "All images must share the same size."
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
            imgpoints.append(corners)

    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    rms, _, _, _, _ = \
        cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            gray.shape[::-1],
            K,
            D,
            rvecs,
            tvecs,
            calibration_flags,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
    # You should replace these 3 lines with the output in calibration step
    DIM = _img_shape[::-1]
    K = np.array(K.tolist())
    D = np.array(D.tolist())

    np.save(mat_path % ("img_points.npy"), np.array(imgpoints))
    np.save(mat_path % ("obj_points.npy"), np.array(objpoints))
    np.save(mat_path%("DIM.npy"), np.array(DIM))
    np.save(mat_path%("K.npy"), K)
    np.save(mat_path%("D.npy"), D)



class FileVideoStream:
    def __init__(self, path, queueSize=128):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(path)
        self.stopped = False

        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queueSize)


def start(self):
    # start a thread to read frames from the file video stream
    t = Thread(target=self.update, args=())
    t.daemon = True
    t.start()
    return self


def update(self):
    # keep looping infinitely
    while True:
        # if the thread indicator variable is set, stop the
        # thread
        if self.stopped:
            return

        # otherwise, ensure the queue has room in it
        if not self.Q.full():
            # read the next frame from the file
            (grabbed, frame) = self.stream.read()

            # if the `grabbed` boolean is `False`, then we have
            # reached the end of the video file
            if not grabbed:
                self.stop()
                return

            # add the frame to the queue
            self.Q.put(frame)


def read(self):
    # return next frame in the queue
    return self.Q.get()

def more(self):
    # return True if there are still frames in the queue
    return self.Q.qsize() > 0

def stop(self):
    # indicate that the thread should be stopped
    self.stopped = True


from imutils.video import FileVideoStream

imgpoints = np.load(mat_path % ("img_points.npy"))
objpoints = np.load(mat_path % ("obj_points.npy"))
DIM = np.load(mat_path % ("DIM.npy"))
K = np.load(mat_path % ("K.npy"))
D = np.load(mat_path % ("D.npy"))

map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, tuple(DIM), cv2.CV_16SC2)


def undistort(img):
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img


print("[INFO] starting video file thread...")
fvs = FileVideoStream(1).start()

# loop over frames from the video file stream
while fvs.more():
    t1 = time.time()

    frame = fvs.read()
    frame = imutils.resize(frame, width=1920)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    undistorted_img = undistort(frame)

    t2 = time.time()

    print("FPS : ", 1 / (t2-t1))

    # show the frame and update the FPS counter
    cv2.imshow("Frame", undistorted_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


