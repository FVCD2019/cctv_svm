import cv2
import glob
import numpy as np

images = glob.glob("IPM_samples/IPM_*.jpg")

images = [cv2.imread(img) for img in sorted(images)]

add_AC = cv2.addWeighted(images[0], 0.5, images[2], 0.5, 0)
add_BD = cv2.addWeighted(images[1], 0.5, images[3], 0.5, 0)
add_all = cv2.addWeighted(add_AC, 0.5, add_BD, 0.5, 0)

cv2.imwrite("/home/siit/stitching/A.jpg", cv2.resize(images[0], (1920, 1080)))
cv2.imwrite("/home/siit/stitching/B.jpg", cv2.resize(images[1], (1920, 1080)))
cv2.imwrite("/home/siit/stitching/C.jpg", cv2.resize(images[2], (1920, 1080)))
cv2.imwrite("/home/siit/stitching/D.jpg", cv2.resize(images[3], (1920, 1080)))

cv2.imwrite("/home/siit/stitching/add_AC.jpg", cv2.resize(add_AC, (1920, 1080)))
cv2.imwrite("/home/siit/stitching/add_BD.jpg", cv2.resize(add_BD, (1920, 1080)))
cv2.imwrite("/home/siit/stitching/add_all.jpg", cv2.resize(add_all, (1920, 1080)))
