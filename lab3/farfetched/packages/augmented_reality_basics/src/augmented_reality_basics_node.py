#!/usr/bin/env python3
import os
import rospy
import time
import yaml

import cv2
import numpy as np

from duckietown_msgs.srv import SetCustomLEDPattern, ChangePattern
from duckietown_msgs.srv import SetCustomLEDPatternResponse, ChangePatternResponse
from duckietown_msgs.msg import LEDPattern
from std_msgs.msg import ColorRGBA

from duckietown.dtros import DTROS, TopicType, NodeType
from augmented_reality_basics import Augmenter

from augmented_reality_basics import Augmenter

# In the ROS node, you just need a callback on the camera image stream that
# uses the Augmenter class to modify the input image. Therefore, implement
# a method called callback that writes the augmented image to the appropriate
# topic.

# Load the intrinsic / extrinsic calibration parameters for the given robot.
# Read the map file corresponding to the map_file parameter given in the roslaunch
# command above.
# Subscribe to the image topic /robot name/camera_node/image/compressed.
# When you receive an image, project the map features onto it, and then publish
# the result to the topic /robot name/node_name/map file basename/image/compressed
# where map file basename is the basename of the file without the yaml extension.

class ARBasicsNode(DTROS):
    def __init__(self, node_name):
        super(ARBasicsNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.DRIVER)

        self.augmenter = Augmenter()
        self.hostname = rospy.get_param("~veh")

        # Read in yaml ====
        yaml_file = rospy.get_param("~map_file")

        with open(yaml_file, 'r') as y:
            self.map = yaml.load(y, Loader=yaml.CLoader)

        # Setup publisher path ====
        yaml_basename = os.path.basename(os.path.splitext(p)[0])

        self.pub = rospy.Publisher(
            f"/{self.hostname}/node_name/{yaml_basename}/image/compressed",
            CompressedImage,
            queue_size=2,
        )

        self.img_sub(f"/{self.hostname}/camera_node/image/compressed",
                     self.callback_image)

    def callback_image(self, compressed):
        """Callback for the image topic."""
        raw_bytes = np.frombuffer(compressed.data, dtype=np.uint8)
        cv_img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)

        self.image = self.augmenter.render_segments(cv_img, self.map)

    def on_shutdown(self):
        """Shutdown procedure.

        At shutdown, changes the LED pattern to `LIGHT_OFF`.
        """
        # Turn off the lights when the node dies
        self.loginfo("Shutting down. Turning LEDs off.")
        self.changePattern("LIGHT_OFF")
        time.sleep(1)


if __name__ == "__main__":
    ar_node = ARBasicsNode(node_name="augmented_reality_basics_node")

    rate = rospy.Rate(1)

    while not rospy.is_shutdown():
        if ar_node.image is not None:
            ar_node.pub(ar_node.image)
        rate.sleep()

    rospy.spin()

# Load the intrinsic / extrinsic calibration parameters for the given robot.
# Read the map file corresponding to the map_file parameter given in the roslaunch command above.
# Subscribe to the image topic /robot name/camera_node/image/compressed.
# When you receive an image, project the map features onto it, and then publish the result to the topic /robot name/node_name/map file basename/image/compressed where map file basename is the basename of the file without the yaml extension.
# Create a ROS node called augmented_reality_basics_node, which imports an Augmenter class, from an augmented_reality_basics module. The Augmenter class should contain the following methods:
