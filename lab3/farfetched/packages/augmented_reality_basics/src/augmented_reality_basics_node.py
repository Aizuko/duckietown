#!/usr/bin/env python3
import time
import rospy
import yaml

from duckietown_msgs.srv import SetCustomLEDPattern, ChangePattern
from duckietown_msgs.srv import SetCustomLEDPatternResponse, ChangePatternResponse
from duckietown_msgs.msg import LEDPattern
from std_msgs.msg import ColorRGBA

from duckietown.dtros import DTROS, TopicType, NodeType

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
        super(ARBasicsNode, self)
            .__init__(node_name=node_name, node_type=NodeType.DRIVER)

        yaml_file = rospy.get_param("~map_file")

        with open(yaml_file, 'r') as y:
            self.config = yaml.load(y, Loader=yaml.CLoader)

        self.img_sub(f"/{hostname}/camera_node/image/compressed",
                     self.cb_input_image)

    def cb_input_image(self, msg):
        pass

    def on_shutdown(self):
        """Shutdown procedure.

        At shutdown, changes the LED pattern to `LIGHT_OFF`.
        """
        # Turn off the lights when the node dies
        self.loginfo("Shutting down. Turning LEDs off.")
        self.changePattern("LIGHT_OFF")
        time.sleep(1)


if __name__ == "__main__":
    led_emitter_node = ARBasicsNode(node_name="augmented_reality_basics_node")
    rospy.spin()

# Load the intrinsic / extrinsic calibration parameters for the given robot.
# Read the map file corresponding to the map_file parameter given in the roslaunch command above.
# Subscribe to the image topic /robot name/camera_node/image/compressed.
# When you receive an image, project the map features onto it, and then publish the result to the topic /robot name/node_name/map file basename/image/compressed where map file basename is the basename of the file without the yaml extension.
# Create a ROS node called augmented_reality_basics_node, which imports an Augmenter class, from an augmented_reality_basics module. The Augmenter class should contain the following methods:
