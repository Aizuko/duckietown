#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------


# NOTE: Use the variable DT_REPO_PATH to know the absolute path to your code
# NOTE: Use `dt-exec COMMAND` to run the main process (blocking process)

# launching app
# dt-exec roslaunch augmented_reality_basics augmented_reality_basics.launch map_file:="maps/calibration_pattern.yaml" veh:="$VEHICLE_NAME"
#dt-exec roslaunch apriltag apriltag.launch veh:="$VEHICLE_NAME"
roscore&
sleep 5
rosbag play /data/bag.bag --loop&
dt-exec roslaunch lane_finder lane_finder.launch veh:="csc22906"
#dt-exec roslaunch lane_follower lane_follower.launch veh:="$VEHICLE_NAME"

# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE

# wait for app to end
dt-launchfile-join
