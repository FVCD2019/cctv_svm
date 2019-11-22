import numpy as np
import cv2
import math
from PIL import Image
import random
import os
import copy
import matplotlib.pyplot as plt


# final image cooridnate

k = np.pi/180
trans = np.array([[650], [600]])
crop = np.array([[300],[600]]) # the crop size
car1 = np.array([[679], [351]]) # car center1
car2 = np.array([[650], [370]]) # car center2
new_cen = np.array([0,0])
theta1 = 356 # Heading angle
theta2 = 356
Rotation = np.array([[math.cos((360 - theta1)*k), -math.sin((360 - theta1)*k)],[math.sin((360 - theta1)*k), math.cos((360 - theta1)*k)]])
Rotation2 = np.array([[math.cos((360 - theta2)*k), -math.sin((360 - theta2)*k)],[math.sin((360 - theta2)*k), math.cos((360 - theta2)*k)]])

cen_trans = car1 - trans # translation image 의 중심을 원점(0,0)으로 했을 때 차량 중심의 좌표 (y축은 image coordinate 방향을 따름)
cen_trans2 = car2 - trans
print("cen_trans", cen_trans)

cen_rot = np.round((np.matmul(Rotation, cen_trans)), 0) # 정수로 반올림
cen_rot2 = np.round((np.matmul(Rotation2, cen_trans2)), 0)

cen_rotated = cen_rot + 2*trans
cen_rotated2 = cen_rot2 + 2*trans
area1 = (int(cen_rotated[0]-crop[0]/2), int(cen_rotated[1]-crop[1]/2), int(crop[0]/2), int(crop[1])) # crop할 이미지의 영역
area2 = (int(cen_rotated2[0]), int(cen_rotated2[1]-crop[1]/2), int(crop[0]/2), int(crop[1]))
print("crop area1:", area1)
print("crop area2:", area2)

img1 = cv2.imread("C:\\Users\\user\\Desktop\\ipm0_1120.png") # 1번쨰 카메라 이미지
plt.imshow(img1)
plt.show()
exit()

img2 = cv2.imread("C:\\Users\\user\\Desktop\\ipm1_1120.png") # 2번쨰 카메라 이미지
img_temp = cv2.imread("C:\\Users\\user\\Desktop\\ipm0_1120.png") # 빈 이미지를 만들기 위한 작업
'''
ggg = img1.copy()
ggg = img1[0:400, 0:1200] # 앞에가 y좌표(100부터 400) 뒤에가 x좌표(100부터 1200)
cv2.imshow("ggg",ggg)
cv2.waitKey()
exit()
'''

rows, cols = img1.shape[:2]
M1 = np.float32([[1,0,650],[0,1,600]]) # translation x+650, y+600
img_trans1 = cv2.warpAffine(img1, M1, (2*cols, 2*rows)) # img_trans는 img를 M1만큼 translation시킨 것

M1 = cv2.getRotationMatrix2D((1300, 1200), theta1-360, 1) # rotation center is x=1300, y=1200
dst1 = cv2.warpAffine(img_trans1, M1, (2*cols, 2*rows))

dst1_copy = dst1.copy()
dst1_copy = dst1[area1[1]:area1[1]+area1[3], area1[0]:area1[0]+area1[2]] # 앞에께 세로 뒤에께 가로



M2 = np.float32([[1,0,650],[0,1,600]]) # translation x+650, y+600
img_trans2 = cv2.warpAffine(img2, M2, (2*cols, 2*rows)) # img_trans는 img를 M2만큼 translation시킨 것

M2 = cv2.getRotationMatrix2D((1300, 1200), theta2-360, 1) # rotation center is x=1300, y=1200
dst2 = cv2.warpAffine(img_trans2, M2, (2*cols, 2*rows))

dst2_copy = dst2.copy()
dst2_copy = dst2[area2[1]:area2[1]+area2[3], area2[0]:area2[0]+area2[2]] # 앞에께 세로 뒤에께 가로

final = cv2.hconcat([dst1_copy, dst2_copy]) # 이미지 두개 가로로 붙이기

final[160:440, 60:240, :] = 0

cv2.imshow("final", final)
cv2.waitKey()
