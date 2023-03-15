# ucb_player2

This workspace implements the exercise 4 requirements. Namely,
it implements a PID controller, computer vision for lane following, computer
vision for grid detection on the back of a duckiebot, and a finite state machine
for safely tracking a duckiebot across duckietown.

This ROS program is based on
[Xiao's](https://github.com/XZPshaw/CMPUT412503_exercise4) grid-detection
template, with contributions from the lane following package Justin Francis
posted on eclass.

## Installation and Usage

```bash
docker -H csc22902.local build -t duckietown/duckietown:latest-arm64v8 .
dts devel run -H csc22902.local -v /data:/data
```

Replace `csc22902` with the hostname of your Duckiebot.

## Packages

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
