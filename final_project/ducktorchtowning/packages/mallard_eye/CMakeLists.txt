cmake_minimum_required(VERSION 2.8.3)
project(mallard_eye)

find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  std_msgs
  message_generation
)

add_service_files(
  DIRECTORY srv
  FILES
  MallardEyedentify.srv
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
