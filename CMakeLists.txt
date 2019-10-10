cmake_minimum_required(VERSION 2.8.3)
project(cctv_svm)

find_package(catkin REQUIRED COMPONENTS
#  opencv
  cv_bridge
  image_transport
  rospy
  sensor_msgs
  std_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES cctv_svm
#  CATKIN_DEPENDS OpenCV cv_bridge image_transport rospy sensor_msgs std_msgs
#  DEPENDS system_lib
)

include_directories(
 include
  ${catkin_INCLUDE_DIRS}
)

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )


