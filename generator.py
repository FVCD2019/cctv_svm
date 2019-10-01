import os
import cv2
import numpy as np


def load_img(path):
    img_vehicle = cv2.imread(path)
    vehicle_row, vehicle_col = img_vehicle.shape[:2]

    # padding for rotation
    img_vehicle_pad = np.ones( (vehicle_row*2, vehicle_col*2, 3), dtype=np.uint8) * 255
    img_vehicle_pad[int(vehicle_row*0.5):int(vehicle_row*1.5) , int(vehicle_col*0.5):int(vehicle_col*1.5), :] = img_vehicle

    return img_vehicle_pad


def vehicle_generator(max_vehicle=2):

    """ load vehicle images"""
    img_vehicles = ["car_bird_eye_view/car%d.jpg" % i for i in (3, 7, 1, 8, 10)]
    #img_vehicles = ["car_bird_eye_view/car%d.jpg" % i for i in range(1, 11)]

    """ load fg & bg image """
    img_background = cv2.imread("perspective_background.jpg")  # 배경영상 불러오기
    num_vehicle = np.random.randint(1, max_vehicle+1)

    """ iou check to prevent overlapping images """
    iou = np.zeros(img_background.shape[:2])
    prev_iou_sum = 0

    for v in range(num_vehicle):
        """ load random vehicle image"""
        vehicle_n = np.random.randint(0, len(img_vehicles))
        img_vehicle_pad = load_img(img_vehicles[vehicle_n])
        # cv2.imshow("pad", img_vehicle_pad)

        """ vehicle random rotation """
        angle = np.random.randint(0, 360)

        vehicle_row, vehicle_col = img_vehicle_pad.shape[:2]
        M = cv2.getRotationMatrix2D((vehicle_col / 2, vehicle_row / 2), angle, 1.0)

        img_vehicle_rotated = cv2.warpAffine(img_vehicle_pad, M, (vehicle_col, vehicle_row),
                                             borderValue=(255, 255, 255))

        """ vehicle resize """
        vehicle_row, vehicle_col  = 1000, 500

        img_vehicle_rotated = cv2.resize(img_vehicle_rotated, dsize=(vehicle_col, vehicle_row),
                                         interpolation=cv2.INTER_LINEAR)

        """ vehicle random location """
        height_bg, width_bg = img_background.shape[:2]
        height_sm, width_sm = img_vehicle_rotated.shape[:2]

        x_offset_s = np.random.randint(0, width_bg - width_sm)
        y_offset_s = np.random.randint(0, height_bg - height_sm)

        """ vehicle tight cropping """
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
        if iou.sum() - prev_iou_sum != (y_offset_s - y_offset_e) * (x_offset_s - x_offset_e):
            # if overlap , then not composition
            num_vehicle -= 1
            continue
        prev_iou_sum = iou.sum()

        img_add_vehicle[y_offset_s:y_offset_e, x_offset_s:x_offset_e] = np.where(img_vehicle_crop < 230
                                                                                 , img_vehicle_crop,
                                                                                 img_background[y_offset_s:y_offset_e,
                                                                                 x_offset_s:x_offset_e])
    return img_add_vehicle


def space_generator(max_vehicle, upper_space, lower_space):

    """ load vehicle images"""
    img_vehicles = ["car_bird_eye_view/car%d.jpg" % i for i in (3, 7, 1, 8, 10)]
    #img_vehicles = ["car_bird_eye_view/car%d.jpg" % i for i in range(1, 11)]

    """ load fg & bg image """
    img_background = cv2.imread("perspective_background.jpg")  # 배경영상 불러오기
    num_vehicle = np.random.randint(1, max_vehicle + 1)

    space = upper_space + lower_space

    """ draw line """
    for s, e in upper_space:
        img_background = cv2.rectangle(img_background, s, e, (255, 0, 0), 3)

    for s, e in lower_space:
        img_background = cv2.rectangle(img_background, s, e, (255, 0, 0), 3)

    """ space image generate """
    for v in range(num_vehicle):
        """ get random vehicle and space"""
        vehicle_n = np.random.randint(0, len(img_vehicles))
        space_n = np.random.randint(0, len(space))

        img_vehicle = load_img(img_vehicles[vehicle_n])

        """ vehicle resize """
        vehicle_row, vehicle_col = 1400, 700

        img_vehicle = cv2.resize(img_vehicle, dsize=(vehicle_col, vehicle_row),
                                 interpolation=cv2.INTER_LINEAR)

        """ pre-defined paking space location """
        x_offset_s = space[space_n][0][0] + 30
        y_offset_s = space[space_n][0][1] + 30

        """ vehicle tight cropping """
        img_border = (img_vehicle < 230).sum(2) == 3

        h_arr, w_arr = np.nonzero(img_border)

        vehicle_h_min, vehicle_h_max = h_arr.min(), h_arr.max()
        vehicle_w_min, vehicle_w_max = w_arr.min(), w_arr.max()

        img_vehicle_crop = img_vehicle[vehicle_h_min:vehicle_h_max, vehicle_w_min:vehicle_w_max, :]

        """ natural composition """
        img_add_vehicle = img_background

        y_offset_e = y_offset_s + vehicle_h_max - vehicle_h_min
        x_offset_e = x_offset_s + vehicle_w_max - vehicle_w_min

        img_add_vehicle[y_offset_s:y_offset_e, x_offset_s:x_offset_e] = np.where(img_vehicle_crop < 230,
                                                                                 img_vehicle_crop,
                                                                                 img_background[y_offset_s:y_offset_e,
                                                                                 x_offset_s:x_offset_e])
    return img_add_vehicle


if __name__ == '__main__':

    if not os.path.exists("samples"):
        os.mkdir("samples")

    num_images = 10

    for compose in range(num_images):

        print("[%03d / %03d]" % (compose, num_images) , end='\r')

        synthetic_image = vehicle_generator()

        cv2.imwrite("samples/sample_%d.jpg" % compose, synthetic_image )

        synthetic_image = cv2.resize(synthetic_image , (1200, 800))

        cv2.imshow("out", synthetic_image)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    print("\ndone")
