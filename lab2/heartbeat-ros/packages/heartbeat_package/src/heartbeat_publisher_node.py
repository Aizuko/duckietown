#!/usr/bin/env python3
import os
import rospy
from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import String

from subprocess import Popen, PIPE

hostname_cmd = Popen(["hostname"], stdout=PIPE);
hostname, _ = hostname_cmd.communicate(timeout=1)

if not hostname:
    hostname = os.environ['VEHICLE_NAME']

class MyPublisherNode(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(MyPublisherNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        # construct publisher
        self.pub = rospy.Publisher('~heartbeat', String, queue_size=10)

    def run(self):
        # publish message every 50s roughly
        rate = rospy.Rate(0.02)  # Measured in Hz
        while not rospy.is_shutdown():
            heartbeat = f"Heartbeat from {hostname}"
            rospy.loginfo(f"Publishing: '{heartbeat}'")
            self.pub.publish(heartbeat)
            rate.sleep()

if __name__ == '__main__':
    # create the node
    node = MyPublisherNode(node_name='heartbeat_publisher')
    # run node
    node.run()
    # keep spinning
    rospy.spin()
