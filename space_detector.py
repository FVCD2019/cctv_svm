import os
import time
import cv2
import numpy as np
from sklearn.externals import joblib

cv2.setNumThreads(1)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter('space_output.avi', fourcc, 1.0, (600,600))

num = 1

classifier = joblib.load('saved_model_3.pkl')
classes = {0:'Empty',1:'Up',2:'Down'}

def Detector(image, space, up=True):
    _image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for s in space:
        (x1, y1), (x2, y2) = s
        space_crop = _image[y1:y2, x1:x2]
        space_crop = cv2.resize(space_crop, (40, 80))

        """ Classify """
        input_x = space_crop.flatten()

        out = classifier.predict(input_x[None, :])
        print(classes[out[0]])

        text = "Empty" if out[0]==0 else "Occupy"

        color = (0, 255, 0) if out[0] == 0 else (0, 0, 255)

        if up:
            image = cv2.rectangle(image, (x1 + 50, y2 + 20), (x1 + 320, y2 + 120), color, -1)
            image = cv2.putText(image, text, (x1 + 80, y2 + 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), thickness=4)
        else:
            image = cv2.rectangle(image, (x1 + 50, y1 - 60), (x1 + 320, y1 - 160), color, -1)
            image = cv2.putText(image, text, (x1 + 80, y1 - 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), thickness=4)

    return image


def load_img(path):
    img_vehicle = cv2.imread(path)
    vehicle_row, vehicle_col = img_vehicle.shape[:2]

    img_vehicle_pad = np.ones( (vehicle_row*2, vehicle_col*2, 3), dtype=np.uint8) * 255
    img_vehicle_pad[int(vehicle_row*0.5):int(vehicle_row*1.5) , int(vehicle_col*0.5):int(vehicle_col*1.5), :] = img_vehicle

    return img_vehicle_pad


num_images = 50
num_vehicle = 2

img_vehicles = ["car_bird_eye_view/car%d.jpg" % i for i in range(1, 11)]

for compose in range(num_images): # 몇 장이나 합성을 할 것인지 설정

    print("[%03d / %03d]" % (compose, num_images) , end='\r')    

    """ load fg & bg image """
    img_background = cv2.imread("perspective_background.jpg")  # 배경영상 불러오기
    num_vehicle = np.random.randint(1, 4) #4)

    """ define space location """
    upper_space = [ [(100+400*i, 100), (500+400*i, 900)] for i in range(6) ] 
    lower_space = [ [(100+400*i, 2200), (500+400*i, 3000)] for i in range(4) ]
    space = upper_space + lower_space


    """ draw line """
    for s, e in upper_space:
        img_background = cv2.rectangle(img_background, s, e, (255, 0, 0), 3)    

    for s, e in lower_space:
        img_background = cv2.rectangle(img_background, s, e, (255, 0, 0), 3)


    for v in range(num_vehicle):
        vehicle_n = np.random.randint(0, len(img_vehicles))
        space_n = np.random.randint(0, len(space))

        img_vehicle = load_img(img_vehicles[vehicle_n])

        """ target resize """
        vehicle_row = 1400    # 연기소스를 resize 시킬 세로 길이 범위설정
        vehicle_col = 700    # 연기소스를 resize 시킬 가로 길이

        img_vehicle = cv2.resize(img_vehicle, dsize = (vehicle_col, vehicle_row), interpolation=cv2.INTER_LINEAR) # 연기소스를 위에서 랜덤하게 설정한 가로 세로 길이로 resize


        """ target tight cropping """
        height_bg, width_bg = img_background.shape[:2]  # 배경영상의 세로 가로 길이 할당
        height_sm, width_sm = img_vehicle.shape[:2]       # 연기소스의 세로 가로 길이 할당

        x_offset_s = space[space_n][0][0] + 30
        y_offset_s = space[space_n][0][1] + 30

        # for cropping
        img_border = (img_vehicle < 230).sum(2) == 3

        h_arr, w_arr = np.nonzero(img_border)
    
        vehicle_h_min, vehicle_h_max = h_arr.min(), h_arr.max()
        vehicle_w_min, vehicle_w_max = w_arr.min(), w_arr.max()

        img_vehicle_crop = img_vehicle[vehicle_h_min:vehicle_h_max, vehicle_w_min:vehicle_w_max, :]


        """ natural composition """
        img_add_vehicle = img_background

        y_offset_e = y_offset_s + vehicle_h_max - vehicle_h_min
        x_offset_e = x_offset_s + vehicle_w_max - vehicle_w_min

        img_add_vehicle[y_offset_s:y_offset_e, x_offset_s:x_offset_e] = np.where( img_vehicle_crop < 230 , img_vehicle_crop, img_background[y_offset_s:y_offset_e, x_offset_s:x_offset_e] )


    cv2.imshow("input", cv2.resize(img_add_vehicle, (600, 600)))
    """ vehicle detect """
    t0 = time.time()
    img_detected = Detector(img_add_vehicle, upper_space, up=True)
    img_detected = Detector(img_detected, lower_space, up=False)
    t1 = time.time()

    img_detected = cv2.resize(img_detected, (600, 600))
    img_detected = cv2.putText(img_detected, "FPS : %d" % (1/(t1-t0)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), thickness=3) 

    cv2.imshow("out", img_detected)
    if compose > 0:
        video_writer.write(img_detected)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
video_writer.release()

print("\ndone")

