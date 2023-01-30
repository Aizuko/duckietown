#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

hostname = os.environ['VEHICLE_NAME']

class MySubscriberNode(DTROS):
    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(MySubscriberNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        self.pub_info = rospy.Publisher(
            '~path_to/published_image_info',
            String,
            queue_size=2,
        )
        self.pub_comp = rospy.Publisher(
            '~path_to/published_compressed',
            CompressedImage,
            queue_size=2,
        )
        self.sub_info = rospy.Subscriber(
            f'~path_to/camera_info',
            CameraInfo,
            self.callback_info,
        )
        self.sub_comp = rospy.Subscriber(
            f'~path_to/camera_compressed',
            CompressedImage,
            self.callback_image,
        )

    def callback_info(self, img_info):
        img_dims = f"{img_info.height}x{img_info.width}"
        rospy.loginfo(f"Image size: {img_dims}")
        self.pub_info.publish(img_dims)

    def callback_image(self, compressed):
        rospy.loginfo("Republishing image")
        self.pub_comp.publish(compressed)

if __name__ == '__main__':
    # create the node
    node = MySubscriberNode(node_name='compressed')
    # keep spinning
    rospy.spin()
