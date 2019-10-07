import argparse
import numpy as np
from network.resnet import *
import cv2
import matplotlib.pyplot as plt
import scipy.misc as m
import time
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
#assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np
import os
import glob
import sys
import matplotlib.pyplot as plt
mat_path = "D:/19년 2학기/창시구/calibration/%s"

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
    images = glob.glob('C:/Users/user/Desktop/check/Webcam/Webcam/*.jpg')
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
    print("Found " + str(N_OK) + " valid images for calibration")
    print("DIM=" + str(_img_shape[::-1]))
    print("K=np.array(" + str(K.tolist()) + ")")
    print("D=np.array(" + str(D.tolist()) + ")")

    # You should replace these 3 lines with the output in calibration step
    DIM = _img_shape[::-1]
    K = np.array(K.tolist())
    D = np.array(D.tolist())

    np.save(mat_path % ("img_points.npy"), np.array(imgpoints))
    np.save(mat_path % ("obj_points.npy"), np.array(objpoints))
    np.save(mat_path%("DIM.npy"), np.array(DIM))
    np.save(mat_path%("K.npy"), K)
    np.save(mat_path%("D.npy"), D)

imgpoints = np.load(mat_path % ("img_points.npy"))
objpoints = np.load(mat_path % ("obj_points.npy"))
DIM = np.load(mat_path % ("DIM.npy"))
K = np.load(mat_path % ("K.npy"))
D = np.load(mat_path % ("D.npy"))

print("DIM=",DIM)
print("K:",K)
print("D:",D)

def undistort(img_path):
    img = cv2.imread(img_path)
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, tuple(DIM), cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    undistorted_img = cv2.resize(undistorted_img, (1280, 720))
    plt.imshow(undistorted_img)
    plt.show()
    #cv2.imshow("undistorted", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#if __name__ == '__main__':

undistort("C:/Users/user/Desktop/tiles/tile3.jpg")



'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("/home/siit/Desktop/tile.jpg")

pts1 = np.float32([[450,150],[10,1900],[3000,120],[3300,1900]])

# 좌표의 이동점
pts2 = np.float32([[10,10],[10,3000],[3000,10],[3000,3000]])

# pts1의 좌표에 표시. perspective 변환 후 이동 점 확인.
cv2.circle(img, (450,150), 50, (255,0,0),-1)
cv2.circle(img, (10,1900), 50, (0,255,0),-1)
cv2.circle(img, (3000,120), 50, (0,0,255),-1)
cv2.circle(img, (3300,1900), 50, (0,0,0),-1)

M = cv2.getPerspectiveTransform(pts1, pts2)

dst = cv2.warpPerspective(img, M, (3100,3100))

plt.imshow(img),plt.title('image')
plt.show()
plt.imshow(dst),plt.title('Perspective')
plt.show()
'''
