# Exercise 5 Lab Code

## mallardvision

This directory contains the ROS workspace for this lab. It contains the
apriltag, deadreckoning, lane_follower, and mallard_eye nodes which implement
the requirements for Deliverable 3 in this lab.

## pytorch

This directory contains the raw data, scripts for preprocessing, training,
evaluating a simple multilayer perceptron using PyTorch. The weights learned
here are used in the `mallard_eye` node.

## Multilayer_Perceptron.ipynb

This notebook was used to fulfill the requirements of Deliverable 2. We used
the [MPS backend](https://pytorch.org/docs/stable/notes/mps.html) available on
MacOS devices as the GPU resource in our runs.
