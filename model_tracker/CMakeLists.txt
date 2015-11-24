cmake_minimum_required(VERSION 2.8.3)
project(model_tracker)
find_package(catkin REQUIRED COMPONENTS  
  message_generation
  std_msgs
  sensor_msgs
  stereo_msgs
  roscpp
  pcl_ros
  pcl_conversions
  tf
  eigen_conversions
  dynamic_reconfigure
  tf_conversions)
  
# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
set(ROS_BUILD_TYPE Debug)

link_directories(${catkin_LIBRARY_DIRS})

find_package(PCL 1.8 REQUIRED)
include_directories(include ${catkin_INCLUDE_DIRS}  ${PCL_INCLUDE_DIRS})

FIND_PACKAGE(Boost COMPONENTS filesystem thread REQUIRED )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

find_package(Qt4 COMPONENTS QtCore QtGui REQUIRED)
include(${QT_USE_FILE})

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
add_definitions(-DQT_NO_KEYWORDS)
add_definitions(-std=c++11)

add_message_files(
  FILES
  ModelTracker.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  sensor_msgs
  stereo_msgs
)


catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS message_runtime pcl_ros pcl_conversions tf eigen_conversions tf_conversions)
 
add_executable(model_tracker_norecog src/model_tracker_norecog.cpp)
add_dependencies(model_tracker_norecog ${catkin_EXPORTED_TARGETS})
target_link_libraries(model_tracker_norecog ${catkin_LIBRARIES} ${VTK_LIBRARIES} ${PCL_LIBRARIES} ${BOOST_LIBRARIES}) 
