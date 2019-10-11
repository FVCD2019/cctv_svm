import cv2
import time
import numpy as np
from generator import vehicle_generator
from detector import Vehicle_Detector

""" video writer """
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter('vehicle_output.avi', fourcc, 1.0, (600, 600))

num_images = 50
max_vehicle = 2

for compose in range(num_images):

    print("[%03d / %03d]" % (compose, num_images), end='\r')

    """ vehicle image generation"""
    img_vehicle = vehicle_generator(max_vehicle)

    img_vehicle = cv2.resize(img_vehicle, (600, 600))
    gen_img = img_vehicle.copy()
    # cv2.imshow("gen", gen_img)

    """ vehicle detect """
    t0 = time.time()
    img_detected = Vehicle_Detector(img_vehicle)
    t1 = time.time()

    img_detected = cv2.putText(img_detected, "FPS : %d" % (1 / (t1 - t0)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                               (0, 0, 0), thickness=3)

    cv2.imshow("out", img_detected)
    if compose > 0:
        video_writer.write(gen_img)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

print("\ndone")
