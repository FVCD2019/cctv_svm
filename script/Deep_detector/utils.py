import os, cv2
import numpy as np


def get_center_point_contour(output, thresh, scale):
    height, width = output.shape

    mask = (output > thresh).astype(np.uint8)

    results = []

    nLabels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    for k in range(1, nLabels):
        #size = stats[k, cv2.CC_STAT_AREA]

        # make segmentation map
        segmap = np.zeros_like(mask, dtype=np.uint8)
        segmap[labels == k] = 255
        # cv2.dilate(segmap, kernel, segmap)

        im2, contours, hierarchy = cv2.findContours(segmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for cnt in contours:

            rect = cv2.minAreaRect(cnt)

            (rbox_x, rbox_y), (rbox_width, rbox_height), rect_angle = rect

            if rbox_width > rbox_height:
                rbox_width, rbox_height = rbox_height, rbox_width
                rect_angle += 90

            if rbox_y < height//2:
                scale_factor_h = (height-rbox_y)/height/10+1
                scale_factor_w = 1 / ((width-rbox_x)/width/50+1)
            else:
                scale_factor_h = 1 / ((height-rbox_y)/height/10+1)
                scale_factor_w = (width-rbox_x)/width/50+1

            rect = ((scale_factor_w*rbox_x * 2, scale_factor_h*rbox_y*2), (rbox_width * scale * 2, rbox_height * scale * 2), rect_angle)

            box = cv2.boxPoints(rect)

            results.append({"rbox": box, "rect": rect})

    return results


def get_ipm_matrix():
    width, height = 1300, 1200

    #src_pts = np.float32([[660, 170], [1400, 200], [1400, 937], [622, 933]])
    src_pts = np.float32([(0, 0), (width, 0), (width, height), (0, height)])
    dst_pts = np.float32([(0, 0), (width, 0), (width, height), (0, height)])

    IPM_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    return IPM_matrix


def perspective_transform(img, IPM_matrix, width=1300, height=1200):
    img_cuda = cv2.UMat(img)
    img_IPM = cv2.warpPerspective(img_cuda, IPM_matrix, (width, height))

    return img_IPM.get()


def vehicle_crop(image, mask, rect):
    (rbox_x, rbox_y), (rbox_width, rbox_height), rect_angle = rect

    box = cv2.boxPoints(rect)

    """ vehicle cropping using perspective transform (classifier input) """
    width, height = int(rbox_width), int(rbox_height)
    src_pts = box.astype(np.float32)
    dst_pts = np.float32([[0, height], [0, 0], [width, 0], [width, height]])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    img_vehicle_crop = cv2.warpPerspective(image, M, (width, height))
    mask_vehicle_crop = cv2.warpPerspective(mask, M, (width, height))

    """ crop image -90 degree rotation """
    if width > height:
        img_vehicle_crop = cv2.transpose(img_vehicle_crop)
        mask_vehicle_crop = cv2.transpose(mask_vehicle_crop)

        img_vehicle_crop = cv2.flip(img_vehicle_crop, flipCode=0)
        mask_vehicle_crop = cv2.flip(mask_vehicle_crop, flipCode=0)

    """ If output is "down", then angle+=180 """
    if heading_classifier(mask_vehicle_crop):  # if down
        rect_angle += 180

    """ angle re-arrange (0~360) """
    rect_angle = 360 + rect_angle if rect_angle < 0 else rect_angle
    rect_angle = rect_angle - 360 if rect_angle > 360 else rect_angle

    #cv2.imshow("crop", img_vehicle_crop)
    #cv2.imshow("masks", mask_vehicle_crop)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return rect_angle


def vehicle_mask_crop(mask, rect):
    (rbox_x, rbox_y), (rbox_width, rbox_height), rect_angle = rect

    box = cv2.boxPoints(rect)

    """ vehicle cropping using perspective transform (classifier input) """
    width, height = int(rbox_width), int(rbox_height)
    src_pts = box.astype(np.float32)
    dst_pts = np.float32([[0, height], [0, 0], [width, 0], [width, height]])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    mask_vehicle_crop = cv2.warpPerspective(mask, M, (width, height))

    """ crop image -90 degree rotation """
    if width > height:
        mask_vehicle_crop = cv2.transpose(mask_vehicle_crop)

        mask_vehicle_crop = cv2.flip(mask_vehicle_crop, flipCode=0)

    """ If output is "down", then angle+=180 """
    if heading_classifier(mask_vehicle_crop):  # if down
        rect_angle += 180

    """ angle re-arrange (0~360) """
    rect_angle = 360 + rect_angle if rect_angle < 0 else rect_angle
    rect_angle = rect_angle - 360 if rect_angle > 360 else rect_angle

    return rect_angle

def heading_classifier(mask_vehicle_crop):

    m_h, m_w = mask_vehicle_crop.shape

    up_mask = mask_vehicle_crop[:m_h//2, :].max()
    bottom_mask = mask_vehicle_crop[m_h//2:, :].max()

    return True if up_mask < bottom_mask else False


def distance(p1, p2):
    if p2 is None:
        return 0
    x1, y1 = p1
    x2, y2 = p2

    return np.sqrt( (x2-x1)**2 + (y2-y1)**2 )

