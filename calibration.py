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
