import os
import time
import cv2
import numpy as np
from sklearn.externals import joblib

cv2.setNumThreads(1)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter('gen_output.avi', fourcc, 1.0, (600,600))

num = 1

classifier = joblib.load('saved_model_3.pkl')
classes = {0:'Empty',1:'Up',2:'Down'}

def Detector(image):
    global num

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    edged = cv2.Canny(blurred, 80, 200)
    cv2.imshow("edge", edged)

    kernel = np.ones((5,5),np.uint8)
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    _, contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image = cv2.drawContours(image, contours, -1, (0, 0, 255), 3)
    cv2.imshow("contour", image)

    image_temp = image.copy()

    for cnt in contours:

        if not cv2.contourArea(cnt) > 1000:
            continue

        # Straight Rectangle
        #x, y, w, h = cv2.boundingRect(cnt)
        #image = cv2.rectangle(image,(x,y),(x+w, y+h),(0,255,0), 10) # green

        # Rotated Rectangle
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)

        (x, y), (width, height), rect_angle = rect

        #angle = 180+rect_angle if width<height else 90+rect_angle
        angle = 90+rect_angle if width>height else rect_angle

        box = np.int0(box)
        image = cv2.drawContours(image, [box], -1, (0,255,0), 3) # blue
        image_temp = cv2.drawContours(image_temp, [box], -1, (0,255,0), 3) # blue

        """ cropping """
        width, height = int(width), int(height)
        src_pts = box.astype(np.float32)
        dst_pts = np.float32([[0,height], [0,0], [width,0], [width,height]])

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        img_vehicle_crop = cv2.warpPerspective(image, M, (width, height))

        if width > height:
            img_vehicle_crop = cv2.transpose(img_vehicle_crop)
            img_vehicle_crop = cv2.flip(img_vehicle_crop, flipCode=0)

        #img_vehicle_crop = cv2.resize(img_vehicle_crop, (90, 180))
        img_vehicle_crop = cv2.resize(img_vehicle_crop, (40, 80))
        img_vehicle_crop = cv2.cvtColor(img_vehicle_crop,cv2.COLOR_BGR2GRAY)

        """ Classify """
        input_x = img_vehicle_crop.flatten()

        out = classifier.predict(input_x[None, :])

        if out[0] == 2:
            angle += 180

        angle = 360+angle if angle<0 else angle

        image = cv2.putText(image, "heading : %d" % (angle), tuple(box[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), thickness=2) 

        cv2.imshow("crop", img_vehicle_crop)
        cv2.imshow("box", image_temp)
        #cv2.imwrite("DB/crop_%d.jpg" % num, img_vehicle_crop)
        #num += 1

    return image


def load_img(path):
    img_vehicle = cv2.imread(path)
    vehicle_row, vehicle_col = img_vehicle.shape[:2]

    img_vehicle_pad = np.ones( (vehicle_row*2, vehicle_col*2, 3), dtype=np.uint8) * 255
    img_vehicle_pad[int(vehicle_row*0.5):int(vehicle_row*1.5) , int(vehicle_col*0.5):int(vehicle_col*1.5), :] = img_vehicle

    #img_vehicle_pad = np.ones( (int(vehicle_row*2.4), int(vehicle_col*2.4), 3), dtype=np.uint8) * 255
    #img_vehicle_pad[int(vehicle_row*0.7):int(vehicle_row*1.7) , int(vehicle_col*0.7):int(vehicle_col*1.7), :] = img_vehicle

    return img_vehicle_pad


num_images = 10
num_vehicle = 2

img_vehicles = ["car_bird_eye_view/car%d.jpg" % i for i in (3, 7, 1, 8, 10)]

for compose in range(num_images): # 몇 장이나 합성을 할 것인지 설정

    print("[%03d / %03d]" % (compose, num_images) , end='\r')    

    """ load fg & bg image """
    img_background = cv2.imread("perspective_background.jpg")  # 배경영상 불러오기
    num_vehicle = np.random.randint(1, 3) #4)
    iou = np.zeros(img_background.shape[:2])
    prev_iou_sum = 0

    for v in range(num_vehicle):
        vehicle_n = np.random.randint(0, len(img_vehicles))
        img_vehicle_pad = load_img(img_vehicles[vehicle_n])
        #cv2.imshow("pad", img_vehicle_pad)

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

        """ check iou"""
        iou[y_offset_s:y_offset_e, x_offset_s:x_offset_e] = 1
        if iou.sum() - prev_iou_sum != (y_offset_s-y_offset_e) * (x_offset_s-x_offset_e):
            num_vehicle -= 1
            continue
        prev_iou_sum = iou.sum()

        check = (img_vehicle_crop < 230) + (img_vehicle_crop > 20)

        img_add_vehicle[y_offset_s:y_offset_e, x_offset_s:x_offset_e] = np.where( img_vehicle_crop < 230
                                                                                  , img_vehicle_crop,
                                                                                  img_background[y_offset_s:y_offset_e, x_offset_s:x_offset_e] )

    img_add_vehicle = cv2.resize(img_add_vehicle, (600, 600))
    gen_img = img_add_vehicle.copy()
    cv2.imshow("gen", gen_img)

    """ vehicle detect """
    t0 = time.time()
    img_detected = Detector(img_add_vehicle)
    t1 = time.time()

    #img_detected = cv2.resize(img_detected, (1200, 800))
    img_detected = cv2.putText(img_detected, "FPS : %d" % (1/(t1-t0)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), thickness=3)

    #img_detected = cv2.resize(img_detected, (1200, 800))
    cv2.imshow("out", img_detected)
    if compose > 0:
        video_writer.write(gen_img)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

print("\ndone")

