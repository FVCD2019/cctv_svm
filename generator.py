import os
import cv2
import numpy as np

if not os.path.exists("samples"):
    os.mkdir("samples")

num_images = 10

for compose in range(num_images): # 몇 장이나 합성을 할 것인지 설정

    print("[%03d / %03d]" % (compose, num_images) , end='\r')

    """ load fg & bg image """
    img_vehicle = cv2.imread("target.jpg")    # 연기소스 불러오기
    img_background = cv2.imread("perspective_background.jpg")  # 배경영상 불러오기


    """ padding """
    vehicle_row, vehicle_col = img_vehicle.shape[:2]

    img_vehicle_pad = np.ones( (vehicle_row*2, vehicle_col*2, 3), dtype=np.uint8) * 255
    img_vehicle_pad[int(vehicle_row*0.5):int(vehicle_row*1.5) , int(vehicle_col*0.5):int(vehicle_col*1.5), :] = img_vehicle


    """ target random rotation """
    angle = np.random.randint(0, 360)

    vehicle_row, vehicle_col = img_vehicle_pad.shape[:2]
    M = cv2.getRotationMatrix2D( (vehicle_col/2, vehicle_row/2) , angle, 1.0)

    img_vehicle_rotated = cv2.warpAffine(img_vehicle_pad, M, (vehicle_col, vehicle_row), borderValue=(255,255,255))


    """ target resize """
    vehicle_row = 1000    # 연기소스를 resize 시킬 세로 길이 범위설정
    vehicle_col = 500    # 연기소스를 resize 시킬 가로 길이

    img_vehicle_rotated = cv2.resize(img_vehicle_rotated, dsize = (vehicle_col, vehicle_row), interpolation=cv2.INTER_LINEAR) # 연기소스를 위에서 랜덤하게 설정한 가로 세로 길이로 resize


    """ target tight cropping """
    height_bg, width_bg = img_background.shape[:2]  # 배경영상의 세로 가로 길이 할당
    height_sm, width_sm = img_vehicle_rotated.shape[:2]       # 연기소스의 세로 가로 길이 할당

    x_offset_s = np.random.randint(0, width_bg - width_sm)    # 가로길이의 오프셋(합성할 연기의 가로 위치)을 랜덤하게 설정 (최댓값은 연기소스와 배경영상의 가로길이 차이까지)
    y_offset_s = np.random.randint(0, height_bg - height_sm)  # 세로길이의 오프셋(합성할 연기의 세로 위치)을 랜덤하게 설정 (최댓값은 연기소스와 배경영상의 세로길이 차이까지)

    # for cropping
    img_border = (img_vehicle_rotated < 230).sum(2) == 3

    h_arr, w_arr = np.nonzero(img_border)

    vehicle_h_min, vehicle_h_max = h_arr.min(), h_arr.max()
    vehicle_w_min, vehicle_w_max = w_arr.min(), w_arr.max()

    img_vehicle_crop = img_vehicle_rotated[vehicle_h_min:vehicle_h_max, vehicle_w_min:vehicle_w_max, :]


    """ natural composition """
    img_add_vehicle = img_background

    y_offset_e = y_offset_s + vehicle_h_max - vehicle_h_min
    x_offset_e = x_offset_s + vehicle_w_max - vehicle_w_min

    img_add_vehicle[y_offset_s:y_offset_e, x_offset_s:x_offset_e] = np.where( img_vehicle_crop < 230 , img_vehicle_crop, img_background[y_offset_s:y_offset_e, x_offset_s:x_offset_e] )

    cv2.imwrite("samples/sample_%d.jpg" % compose, img_add_vehicle)

    #img_add_vehicle = cv2.resize(img_add_vehicle, (1200, 800))

    #cv2.imshow("out", img_add_vehicle)
    #cv2.waitKey(100)
    #cv2.destroyAllWindows()



print("\ndone")
