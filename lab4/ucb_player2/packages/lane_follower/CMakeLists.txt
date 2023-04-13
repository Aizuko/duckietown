cmake_minimum_required(VERSION 2.8.3)
project(lane_follower)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  nav_msgs
  std_msgs
  # duckietown_msgs # Every duckietown packages should use this.
)

add_dependencies(${catkin_EXPORTED_TARGETS})

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_service_files(
  DIRECTORY srv
  FILES
  StartParking.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
)

catkin_package()
