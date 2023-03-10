#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------


# NOTE: Use the variable DT_REPO_PATH to know the absolute path to your code
# NOTE: Use `dt-exec COMMAND` to run the main process (blocking process)

# launching app
#dt-exec echo "This is an empty launch script. Update it to launch your application."
#roscore &
#sleep 5
#dt-exec rosrun my_package my_publisher_node.py
#dt-exec rosrun my_package my_subscriber_node.py
#roslaunch heartbeat_package heartbeat.launch veh:=$VEHICLE_NAME
#roslaunch camera_demo_node camera.launch veh:=$VEHICLE_NAME
#roslaunch wheel_encoders wheel_encoders.launch veh:=$VEHICLE_NAME
#roslaunch manual_driving manual_driving.launch veh:=$VEHICLE_NAME
dt-exec roslaunch odometry_node odometry.launch veh:=$VEHICLE_NAME
#roslaunch led_controls led_controls.launch veh:=$VEHICLE_NAME

# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE

# wait for app to end
dt-launchfile-join
