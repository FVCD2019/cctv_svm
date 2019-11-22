import os
import cv2
import numpy as np
import glob
from IPM import perspective_transform

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))

    h = h / h.max()

    h[h < np.finfo(h.dtype).eps * h.max()] = 0

    return h

def load_patch(path, longer):
    img_target = cv2.imread(path)
    target_h, target_w\
        = img_target.shape[:2]

    if target_w > target_h:
        re_w = longer
        re_h = int(target_h / target_w * longer)
    else:
        re_h = longer
        re_w = int(target_w / target_h * longer)

    img_target = cv2.resize(img_target , (re_w, re_h))

    return img_target


def vehicle_seg_generator(max_vehicle=2, max_person=2, max_box=4):

    """ load vehicle images """
    img_vehicles = glob.glob("/home/siit/Desktop/Vehicle/DB/DLDB/foreground/car*.png")
    img_persons = glob.glob("/home/siit/Desktop/Vehicle/DB/DLDB/foreground/person*.png")
    img_boxes = glob.glob("/home/siit/Desktop/Vehicle/DB/DLDB/foreground/box*.png")

    """ load fg & bg image """
    img_background = cv2.imread("/home/siit/Desktop/Vehicle/DB/DLDB/background/bg_001.jpg")
    img_background = perspective_transform(img_background)
    img_background = cv2.resize(img_background, (2000, 2000))

    mask_background = np.zeros_like(img_background, dtype=np.uint8)

    num_vehicle = np.random.randint(0, max_vehicle+1)
    num_person = np.random.randint(0, max_person + 1)
    num_box = np.random.randint(0, max_box + 1)

    """ iou check to prevent overlapping images """
    iou = np.zeros(img_background.shape[:2])

    ###################################################################################################################
    for v in range(num_vehicle):
        """ load vehicle images"""
        target_n = np.random.randint(0, len(img_vehicles))
        img_vehicle = cv2.imread(img_vehicles[target_n])

        img_h, img_w, _ = img_vehicle.shape

        """
        diameter = max(img_h, img_w)
        gaussian_map= gaussian2D( (diameter, diameter) ,  sigma=diameter/6)

        mask_vehicle = cv2.resize(gaussian_map, (img_h, img_w), interpolation=cv2.INTER_NEAREST)
        mask_vehicle = np.tile( mask_vehicle[:, :, None]*255, (1,1,3))
        """
        mask_vehicle = np.ones_like(img_vehicle) * 255
        mask_vehicle = mask_vehicle.astype(np.float32)

        for i in range(1, img_h):
            mask_vehicle[i, :, :] -= 128 * i / img_h


        img_vehicle = cv2.resize(img_vehicle, (500, 900))
        mask_vehicle = cv2.resize(mask_vehicle, (500, 900), interpolation=cv2.INTER_NEAREST)

        """ load random vehicle image"""
        # padding for rotation
        vehicle_row, vehicle_col = img_vehicle.shape[:2]
        img_vehicle_pad = np.ones((vehicle_row * 2, vehicle_col * 2, 3), dtype=np.uint8) * 255
        mask_vehicle_pad = np.zeros((vehicle_row * 2, vehicle_col * 2, 3), dtype=np.uint8) * 255

        img_vehicle_pad[int(vehicle_row * 0.5):int(vehicle_row * 1.5), int(vehicle_col * 0.5):int(vehicle_col * 1.5),
        :] = img_vehicle

        mask_vehicle_pad[int(vehicle_row * 0.5):int(vehicle_row * 1.5), int(vehicle_col * 0.5):int(vehicle_col * 1.5),
        :] = mask_vehicle

        """ vehicle random rotation """
        angle = np.random.randint(0, 360)

        vehicle_row, vehicle_col = img_vehicle_pad.shape[:2]
        M = cv2.getRotationMatrix2D((vehicle_col / 2, vehicle_row / 2), angle, 1.0)

        img_vehicle_rotated = cv2.warpAffine(img_vehicle_pad, M, (vehicle_col, vehicle_row),
                                             borderValue=(255, 255, 255))
        mask_vehicle_rotated = cv2.warpAffine(mask_vehicle_pad, M, (vehicle_col, vehicle_row),
                                             borderValue=(0, 0, 0))

        """ vehicle resize """
        vehicle_row, vehicle_col  = 630, 350

        img_vehicle_rotated = cv2.resize(img_vehicle_rotated, dsize=(vehicle_col, vehicle_row),
                                         interpolation=cv2.INTER_LINEAR)
        mask_vehicle_rotated = cv2.resize(mask_vehicle_rotated, dsize=(vehicle_col, vehicle_row),
                                         interpolation=cv2.INTER_NEAREST)

        """ vehicle random location """
        height_bg, width_bg = img_background.shape[:2]
        height_sm, width_sm = img_vehicle_rotated.shape[:2]

        x_offset_s = np.random.randint(0, width_bg - width_sm)
        y_offset_s = np.random.randint(0, height_bg - height_sm)

        """ vehicle tight cropping """
        # for cropping
        #img_border = (img_vehicle_rotated < 230).sum(2) == 3
        img_border = mask_vehicle_rotated.sum(2) > 0

        h_arr, w_arr = np.nonzero(img_border)

        vehicle_h_min, vehicle_h_max = h_arr.min(), h_arr.max()
        vehicle_w_min, vehicle_w_max = w_arr.min(), w_arr.max()

        img_vehicle_crop = img_vehicle_rotated[vehicle_h_min:vehicle_h_max, vehicle_w_min:vehicle_w_max, :]
        mask_vehicle_crop = mask_vehicle_rotated[vehicle_h_min:vehicle_h_max, vehicle_w_min:vehicle_w_max, :]

        """ natural composition """
        y_offset_e = y_offset_s + vehicle_h_max - vehicle_h_min
        x_offset_e = x_offset_s + vehicle_w_max - vehicle_w_min

        """ check iou"""
        if iou[y_offset_s:y_offset_e, x_offset_s:x_offset_e].sum() != 0:
            continue

        iou[y_offset_s:y_offset_e, x_offset_s:x_offset_e] = 1

        img_background[y_offset_s:y_offset_e, x_offset_s:x_offset_e] = np.where(img_vehicle_crop != 255
                                                                                 , img_vehicle_crop,
                                                                                 img_background[y_offset_s:y_offset_e,
                                                                                 x_offset_s:x_offset_e])
        mask_background[y_offset_s:y_offset_e, x_offset_s:x_offset_e] = np.where(mask_vehicle_crop > 0
                                                                                 , mask_vehicle_crop,
                                                                                 mask_background[y_offset_s:y_offset_e,
                                                                                 x_offset_s:x_offset_e])

    ###################################################################################################################
    for v in range(num_person):
        """ load random vehicle image"""
        target_n = np.random.randint(0, len(img_persons))
        img_target = load_patch(img_persons[target_n], 250)
        # cv2.imshow("pad", img_vehicle_pad)

        """ vehicle random location """
        height_bg, width_bg = img_background.shape[:2]
        height_t, width_t = img_target.shape[:2]

        x_offset_s = np.random.randint(0, width_bg - width_t)
        y_offset_s = np.random.randint(0, height_bg - height_t)

        y_offset_e = y_offset_s + height_t
        x_offset_e = x_offset_s + width_t

        """ check iou"""
        if iou[y_offset_s:y_offset_e, x_offset_s:x_offset_e].sum() != 0:
            continue

        iou[y_offset_s:y_offset_e, x_offset_s:x_offset_e] = 1

        img_background[y_offset_s:y_offset_e, x_offset_s:x_offset_e] = img_target

    ###################################################################################################################

    for v in range(num_box):
        """ load random vehicle image"""
        target_n = np.random.randint(0, len(img_boxes))
        img_target = load_patch(img_boxes[target_n], 300)
        # cv2.imshow("pad", img_vehicle_pad)

        """ vehicle random location """
        height_bg, width_bg = img_background.shape[:2]
        height_t, width_t = img_target.shape[:2]

        x_offset_s = np.random.randint(0, width_bg - width_t)
        y_offset_s = np.random.randint(0, height_bg - height_t)

        y_offset_e = y_offset_s + height_t
        x_offset_e = x_offset_s + width_t

        """ check iou"""
        if iou[y_offset_s:y_offset_e, x_offset_s:x_offset_e].sum() != 0:
            continue

        iou[y_offset_s:y_offset_e, x_offset_s:x_offset_e] = 1

        img_background[y_offset_s:y_offset_e, x_offset_s:x_offset_e] = img_target

    return img_background, mask_background


if __name__ == '__main__':

    if not os.path.exists("samples"):
        os.mkdir("samples")

    num_images = 3000

    for compose in range(num_images):

        print("[%03d / %03d]" % (compose, num_images)) # , end='\n')

        synthetic_image, synthetic_mask = vehicle_seg_generator()

        synthetic_image = cv2.resize(synthetic_image, (600, 600))
        synthetic_mask = cv2.resize(synthetic_mask, (600, 600), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite("/home/siit/Desktop/Vehicle/DB/DLDB/train_cont/img/%05d.jpg" % (compose+1), synthetic_image )
        cv2.imwrite("/home/siit/Desktop/Vehicle/DB/DLDB/train_cont/mask/%05d.jpg" % (compose+1), synthetic_mask)

        continue

        synthetic_mask = cv2.applyColorMap(synthetic_mask, cv2.COLORMAP_JET)

        synthetic_image = cv2.resize(synthetic_image , (900, 900))
        synthetic_mask = cv2.resize(synthetic_mask, (900, 900))

        cv2.imshow("out", synthetic_image)
        cv2.imshow("mask", synthetic_mask)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    print("\ndone")
