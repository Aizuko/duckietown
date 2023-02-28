cmake_minimum_required(VERSION 2.8.3)
project(lane_follower)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  cv_bridge
  farfetched_msgs
  duckietown_msgs
)

catkin_package()
