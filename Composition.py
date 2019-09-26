import sys
import cv2
import numpy as np

for compose in range(1): # 몇 장이나 합성을 할 것인지 설정

    img_smoke = cv2.imread("D:/target_rotated.jpg")    # 연기소스 불러오기
    img_background = cv2.imread("D:/perspective_background.jpg")  # 배경영상 불러오기

    smoke_row = 1000    # 연기소스를 resize 시킬 세로 길이 범위설정
    smoke_col = 500    # 연기소스를 resize 시킬 가로 길이

    img_smoke = cv2.resize(img_smoke, dsize = (smoke_col, smoke_row), interpolation=cv2.INTER_LINEAR) # 연기소스를 위에서 랜덤하게 설정한 가로 세로 길이로 resize
    cv2.imshow('a', img_smoke)
    cv2.waitKey(0)
    if img_smoke is None:
        sys.exit()

    height_bg, width_bg = img_background.shape[:2]  # 배경영상의 세로 가로 길이 할당
    height_sm, width_sm = img_smoke.shape[:2]       # 연기소스의 세로 가로 길이 할당

    print(height_bg, width_bg, height_sm, width_sm)

    x_offset = np.random.randint(0, width_bg - width_sm)    # 가로길이의 오프셋(합성할 연기의 가로 위치)을 랜덤하게 설정 (최댓값은 연기소스와 배경영상의 가로길이 차이까지)
    y_offset = np.random.randint(0, height_bg - height_sm)  # 세로길이의 오프셋(합성할 연기의 세로 위치)을 랜덤하게 설정 (최댓값은 연기소스와 배경영상의 세로길이 차이까지)
    print(x_offset, y_offset)


    # for cropping
    smoke_h_max = 0
    smoke_h_min = 9999
    smoke_w_max = 0
    smoke_w_min = 9999
    for a in range(smoke_row):  # resize 시킨 연기 소스를 배경을 제외한 연기 부분만 뽑아서 봤을 때 width의 최소 최대 위치, height의 최소 최대 위치를 구하는 과정
        for b in range(smoke_col):
            if(img_smoke[a,b,0] < 230 and img_smoke[a,b,1] < 230 and img_smoke[a,b,2] < 230): # 합성이미지 픽셀 값 중 모든 픽셀 값이 일정값(250) 이하라면 배경이 아닌 값이므로 
                if(smoke_h_min > a):
                    smoke_h_min = a     # 높이의 최소 값 update
                if(smoke_h_max < a):
                    smoke_h_max = a    # 높이의 최댓 값 update
                if(smoke_w_min > b):
                    smoke_w_min = b     # 너비의 최소 값 update
                if(smoke_w_max < b):
                    smoke_w_max = b     # 너비의 최댓 값 update
    new_img_smoke = np.zeros((smoke_h_max - smoke_h_min, smoke_w_max - smoke_w_min, 3)) # zero 값을 가지는 새로운 3채널 이미지인 new_img_smoke 를 생성해주는데 크기는 위에서 구한 연기소스의 높이와 너비의 최소, 최대 위치 차이만큼 설정

    for row in range(smoke_h_max - smoke_h_min):
        for col in range(smoke_w_max - smoke_w_min):
            for c in range(3):
                new_img_smoke[row, col, c] = img_smoke[row + smoke_h_min, col + smoke_w_min, c] # new_img_smoke 에 crop한 연기를 덮어 씌움

    smoke_weight = np.zeros((smoke_h_max - smoke_h_min, smoke_w_max - smoke_w_min, 3)) # crop한 연기의 크기 만큼 새로운 3채널 이미지 생성
    for i in range(3):
        smoke_weight[:, :, i] = new_img_smoke[:, :, i] / float(240)     # 픽셀값을 240으로 나눠서 0과 1사이의 값으로 만들어서 새로 생성한 이미지에 넣어줌 1에 가까울수록 진한 연기에 가까움


    img_add_smoke = img_background

    for col in range(x_offset, x_offset + smoke_w_max - smoke_w_min):   # transparent 하게 합성하는 과정 (col, row 는 합성이미지의 좌표 값)
        for row in range(y_offset, y_offset + smoke_h_max - smoke_h_min):
            for j in range(3): # 채널 갯수
                if (240*smoke_weight[row - y_offset, col - x_offset, j]) < 230: # 230 보다 작으면 무조건 어떤 물체가 존재한다고 생각하면 됨
                    img_add_smoke[row, col, j] = 240 * smoke_weight[row - y_offset, col - x_offset, j] # 합성할 때 연기의 픽셀값이 얼마나 진한지에 따라 가중치를 고려해서 배경과 합성
                else:
                    img_add_smoke[row, col, j] = img_background[row, col, j] # 흰색바탕 부분은 배경으로 덮어씌움
    cv2.imwrite("D:/composition.jpg", img_add_smoke)
