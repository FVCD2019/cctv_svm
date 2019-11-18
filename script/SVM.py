import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
#x1,y1,angle1
#x2,y2,angle2
#new_x = (x1+x2)/2
#new_y = (y1+y2)/2
#new_angle = (angle1 + angle2)/2
## Headingangle estimation.py --> heading angle, center point

# final image cooridnate
w, h = 360, 720
w ,h = w+1, h+1

x = np.arange(h)
y = np.arange(w)
X, Y = np.meshgrid(x, y)
grid_ori = np.array([list(zip(x, y)) for x, y in zip(X, Y)])
grid_trans = np.array([list(zip(y-w//2, h//2-x)) for x, y in zip(X, Y)])

grid_trans = np.transpose(grid_trans, (1,0,2))
print(grid_trans.shape)
print(grid_trans[400, 300])


w, h = 360, 720
k = np.pi/180

arr_concat = np.array([np.linspace(-w/2, -w/2, h+1), np.linspace(-h/2, h/2, h+1)])
temp = np.array([np.linspace(1, 1, h+1), np.linspace(0, 0, h+1)])
print("Arr:", arr_concat)
print(arr_concat[1,2])
new_concat = arr_concat
for i in range(w):
    new_concat = np.concatenate([new_concat, arr_concat+temp], axis=1)
    arr_concat = arr_concat+temp
    print("new_concat:", new_concat, "i : ", i)

print("---------------")
print(temp)
print(arr_concat)
print(new_concat.shape)
print(new_concat)
theta = 3 # Heading angle from Headingangle_estimation.py

Rotation = np.array([[math.cos(theta*k), -math.sin(theta*k)],[math.sin(theta*k), math.cos(theta*k)]])
car_center = np.zeros((2, (w+1)*(h+1)))
print(car_center.shape)


car_center[0, 0:(w+1)*(h+1)] = 470 # center point x from Headingangle_estimation.py
car_center[1, 0:(w+1)*(h+1)] = 320 # center point y from Headingangle_estimation.py

new_mat = np.matmul(Rotation, new_concat) + car_center
print(new_mat)
new_mat = np.int0(new_mat)
print("-------------")
print("original shape :", new_concat.shape) # new_concat 은 x,y coordinate 에서의 좌표값들 할당
print("new_mat shape:", new_mat.shape) # new_mat 은 변환 후의 이미지 상 coordinate
print("original :", new_concat)
print("new_mat:", new_mat)

img = cv2.imread("C:\\Users\\user\\Desktop\\IPM_cam1.jpg") # 1번쨰 카메라 이미지
img2 = cv2.imread("C:\\Users\\user\\Desktop\\IPM_cam2.jpg") # 2번쨰 카메라 이미지
temp = cv2.imread("C:\\Users\\user\\Desktop\\IPM_cam1.jpg") # 빈 이미지를 만들기 위한 작업
w, h = 360, 720

idx = int((h+1)*(w+2)/2)
print(img.shape)
bg = cv2.resize(temp, (360, 720), interpolation = cv2.INTER_LINEAR)

print("bg:", bg)
print("bg.shape:", bg.shape)

print("new_mat:", new_mat)
print("new_concat:", new_concat)

for i in range(int(w/2)): # cam1 세로로 자른 왼쪽이미지
    bg[0:h, i,:] = img[new_mat[1, 0:h], new_mat[0, i], :] ## 1, i 는 y좌표, 0, i 는 x좌표
    print("img y, x, c : ", img[new_mat[1, 0:h], new_mat[0, i]])
    print("bg :", bg)

for i in range(int(w/2), int(w)): # cam2 세로로 자른 오른쪽 이미지
    bg[0:h, i,:] = img2[new_mat[1, 0:h], new_mat[0, i], :] ## 1, i 는 y좌표, 0, i 는 x좌표
    print("img y, x, c : ", img[new_mat[1, 0:h], new_mat[0, i]])
    print("bg :", bg)

print("bg.shape:", bg.shape)
cv2.imshow('Final SVM image', bg)
cv2.waitKey()


