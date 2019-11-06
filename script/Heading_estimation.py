import cv2
import time
import numpy as np
from generator import vehicle_generator
from detector import Vehicle_Detector

a = np.pi/180




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



camera_cen_x = 651
camera_cen_y = 814
dis = [0,0,0,0]
min_1 = [9999, 0]

for i in range(4):
    dis[i] = math.sqrt((camera_cen_x - rect[i,0])**2+(camera_cen_y - rect[i,1])**2)
    if (dis[i] < min_1[0]):
        min_1[0] = dis[i]
        min_1[1] = i
        
print("rect : ", rect)
print("dis : ", dis)
print("min_1 : ", min_1) # rect[min_1[1]] -> the coordinate of point which is fisrtly closest point from camera center

car_center = [car_center_x, car_center_y] # car center (not revised)
revised_car_center = [0,0]

print("headingangle :", headingangle)
Rotation = np.array([[math.cos(a*headingangle), -math.sin(a*headingangle)],[math.sin(a*headingangle), math.cos(a*headingangle)]])
print("Rotation : ", Rotation)
ori_mat = np.array([[60, 60, -60, -60], [118, -118, 118, -118]])
o_x = rect[min_1[1],0]
o_y = rect[min_1[1],1]
c_x = car_center_x
c_y = car_center_y
ori_point = np.array([[o_x,o_x,o_x,o_x],[o_y,o_y,o_y,o_y]])
print("The minimum distance point is : ", [o_x, o_y])

new_mat = np.matmul(Rotation,ori_mat)+ori_point
print("The 4 candidate center point are : ", new_mat) # 1row is x point, 2row is y point
cen_mat = np.array([[c_x,c_x,c_x,c_x],[c_y,c_y,c_y,c_y]])
dis_mat = new_mat - cen_mat
dis_mat = np.matmul(np.transpose(dis_mat), dis_mat)

min = np.min([dis_mat[0,0],dis_mat[1,1],dis_mat[2,2],dis_mat[3,3]])
idx = 0 # idx is number of index order about candidate 4 center point
for i in range(4):
    if(dis_mat[i,i] == min):
        idx = i

print("car_center : ", car_center_x, car_center_y)
print("revised_car_center is : ", new_mat[0,idx], new_mat[1,idx])
print("GT_car_center : ", 211, 433, "and ", 991.3, 460.1)
print("rect : ", rect)
print("\ndone")
