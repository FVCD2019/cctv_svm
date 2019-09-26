import os
import time
import cv2
import numpy as np


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 2.0, (1200,1000))


def Detector(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edged = cv2.Canny(gray, 80, 150)

    kernel = np.ones((5,5),np.uint8)
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    _, contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:

        if not cv2.contourArea(cnt) > 5000:
            continue

        # Straight Rectangle
        #x, y, w, h = cv2.boundingRect(cnt)
        #image = cv2.rectangle(image,(x,y),(x+w, y+h),(0,255,0), 10) # green

        # Rotated Rectangle
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        image = cv2.drawContours(image, [box], -1, (0,255,0), 5) # blue

    return image


num_images = 30
num_vehicle = 2

for compose in range(num_images): # 몇 장이나 합성을 할 것인지 설정

    print("[%03d / %03d]" % (compose, num_images) , end='\r')

    """ load fg & bg image """
    img_vehicle = cv2.imread("target.jpg")    # 연기소스 불러오기
    img_background = cv2.imread("perspective_background.jpg")  # 배경영상 불러오기


    """ padding """
    vehicle_row, vehicle_col = img_vehicle.shape[:2]

    img_vehicle_pad = np.ones( (vehicle_row*2, vehicle_col*2, 3), dtype=np.uint8) * 255
    img_vehicle_pad[int(vehicle_row*0.5):int(vehicle_row*1.5) , int(vehicle_col*0.5):int(vehicle_col*1.5), :] = img_vehicle

    num_vehicle = np.random.randint(1, 4)

    for v in range(num_vehicle):

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


    img_add_vehicle = cv2.resize(img_add_vehicle, (1200, 1000))


    """ vehicle detect """
    t0 = time.time()
    img_detected = Detector(img_add_vehicle)
    t1 = time.time()

    #img_detected = cv2.resize(img_detected, (1200, 800))
    img_detected = cv2.putText(img_detected, "FPS : %d" % (1/(t1-t0)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), thickness=3) 

    cv2.imshow("out", img_detected)
    out.write(img_detected)

    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

print("\ndone")

