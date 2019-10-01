import cv2
import time
import numpy as np
from generator import space_generator
from detector import Space_Detector

""" video writer """
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter('space_output.avi', fourcc, 1.0, (600, 600))

""" pre-defined space location """
upper_space = [[(100 + 400 * i, 100), (500 + 400 * i, 900)] for i in range(6)]
lower_space = [[(100 + 400 * i, 2200), (500 + 400 * i, 3000)] for i in range(4)]

num_images = 50
max_vehicle = 4

for compose in range(num_images):

    print("[%03d / %03d]" % (compose, num_images), end='\r')

    img_space = space_generator(max_vehicle, upper_space, lower_space)

    cv2.imshow("input", cv2.resize(img_space, (600, 600)))

    """ vehicle detect """
    t0 = time.time()
    img_detected = Space_Detector(img_space, upper_space, up=True)
    img_detected = Space_Detector(img_detected, lower_space, up=False)
    t1 = time.time()

    img_detected = cv2.resize(img_detected, (600, 600))
    img_detected = cv2.putText(img_detected, "FPS : %d" % (1 / (t1 - t0)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                               (0, 0, 0), thickness=3)

    cv2.imshow("out", img_detected)
    if compose > 0:
        video_writer.write(img_detected)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
video_writer.release()

print("\ndone")

