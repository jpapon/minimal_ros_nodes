cmake_minimum_required(VERSION 2.8.3)
project(cnn_classifier)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
)

catkin_python_setup()


catkin_package(
  CATKIN_DEPENDS message_runtime rospy std_msgs sensor_msgs cv_bridge
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

