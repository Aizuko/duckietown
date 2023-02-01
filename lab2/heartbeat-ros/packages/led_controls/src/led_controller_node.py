#!/usr/bin/env python3
# Written by steventango
import os

import numpy as np
import rosbag
import rospy
import time
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import LEDPattern
from std_msgs.msg import ColorRGBA

hostname = os.environ['VEHICLE_NAME']

class LEDControlNode(DTROS):
    """
    Switches the led color?
    """
    def __init__(self, node_name):
        super(LEDControlNode, self).__init__(
            node_name=node_name, node_type=NodeType.CONTROL
        )

        self.pub = rospy.Publisher(
            f'/{hostname}/led_emitter_node/led_pattern',
            LEDPattern,
            queue_size=10,
        )


def create_led_msg(colors: 'List[float]') -> 'LEDPattern':
    """ Creates an led message with the colors set to values from a tuple

    Args:
        Colors (list[float]): A list of 3 floats in the order rgb
    """
    led_msg = LEDPattern()

    for i in range(5):
        rgba = ColorRGBA()
        rgba.r = colors[0]
        rgba.g = colors[1]
        rgba.b = colors[2]
        rgba.a = 1.0
        led_msg.rgb_vals.append(rgba)

    return led_msg


if __name__ == '__main__':
    node = LEDControlNode(node_name='led_controller')
    rospy.loginfo("Starting led controller")

    def turn_off_leds():
        led_msg = create_led_msg([0.0, 0.0, 0.0])
        node.pub.publish(led_msg)
    rospy.on_shutdown(turn_off_leds)

    rate = rospy.Rate(1)

    while not rospy.is_shutdown():
        for rgb in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]:
            led_msg = create_led_msg(rgb)
            node.pub.publish(led_msg)
            rate.sleep()

    turn_off_leds()
