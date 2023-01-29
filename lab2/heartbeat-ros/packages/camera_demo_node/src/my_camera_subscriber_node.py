#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String

hostname = os.environ['VEHICLE_NAME']

class MySubscriberNode(DTROS):
    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(MySubscriberNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        # construct publisher
        self.sub = rospy.Subscriber(
            f'~path_to/camera_info', CameraInfo, self.callback)

    def callback(self, data):
        rospy.loginfo("Saw an image's information?")
        #rospy.loginfo(f"Saw an image of type {type(data.data)}")

if __name__ == '__main__':
    # create the node
    node = MySubscriberNode(node_name='my_camera_subscriber_node')
    # keep spinning
    rospy.spin()
