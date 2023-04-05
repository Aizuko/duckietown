# mallardvision

This workspace implements the exercise 5 requirements. Namely,
it implements a PID controller, computer vision for lane following, machine
learning for digit classification, and a finite state machine for safely
traversing Duckietown in order to identify all correct digits.

This ROS program is based on
[Xiao's](https://github.com/XZPshaw/CMPUT412503_exercise4) grid-detection
template, with contributions from the lane following package Justin Francis
posted on eClass.

## Installation and Usage

```bash
docker -H csc22902.local build -t duckietown/mallardvision:latest-arm64v8 .
dts devel run -H csc22902.local
```

Replace `csc22902` with the hostname of your Duckiebot.

## Packages

### apriltag

This package implements AprilTag detection and localization, it uses the
teleport publisher to update the robot's pose.

### deadreckoning

This package implements dead reckoning with a teleport subscriber that
is used for sensor fusion. This package also publishes any static transforms.

### lane_follower

This package implements the lane follower node, which uses a PID controller
to follow the lane lines. We heavily modified the PID controller, though the
original thing came from Justin Francis's code on eclass.

This node also contains our custom state machines and sensor fusion.

### mallard_creator

This package was used to get training data for the machine learning model.

### mallard_eye

This package implements the number detection node, which uses a multi-layer
perceptron implemented with `numpy` to detect the numbers on blue sticky notes
attached to AprilTags.
