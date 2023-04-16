# ducktorchtowning

This workspace implements the final project requirements. Namely,
it implements a PID controller, computer vision for lane following, computer
vision for grid detection on the back of a duckiebot, apriltag detection,
and a finite state machine for navigating Duckietown.

This ROS program is based on
[Xiao's](https://github.com/XZPshaw/CMPUT412503_exercise4) grid-detection
template, with contributions from the lane following package Justin Francis
posted on eclass.

## Installation and Usage

```bash
docker -H csc22920.local build -t duckietown/ducktorchtowning:latest-arm64v8 .
dts devel run -H csc22920.local -v /data:/data
```

Replace `csc22920` with the hostname of your Duckiebot.

## Packages

### apriltag

This package implements AprilTag detection.

### duckiebot_detection

This one came from the starter code Xiao provided for detecting the grids on the
back on the bot. The original only gave the distance to the bot ahead, though we
modified it to send over the complete transformation matrix to give us angle in
its pose as well.

### lane_follower

This package implements the lane follower node, which uses a PID controller
to follow the lane lines. We heavily modified the PID controller, though the
original thing came from Justin Francis's code on eclass.

This node also contains our custom state machines and sensor fusion. We pretty
much wrote the entire lab in this one file.
