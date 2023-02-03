#!/usr/bin/env python3
import os

import rospy
from duckietown_msgs.msg import WheelsCmdStamped, BoolStamped, WheelEncoderStamped

from duckietown.dtros import DTROS, TopicType, NodeType


class MockWheelsDriverNode(DTROS):
    """Node handling the motor velocities communication.

    Subscribes to the requested wheels commands (linear velocities, i.e. velocity for the left
    and the right wheels) and to an emergency stop flag.
    When the emergency flag `~emergency_stop` is set to `False` it actuates the wheel driver
    with the velocities received from `~wheels_cmd`. Publishes the execution of the commands
    to `~wheels_cmd_executed`.

    The emergency flag is `False` by default.

    Subscribers:
       /{hostname}/wheels_driver_node/wheels_cmd (:obj:`WheelsCmdStamped`): The requested wheel command
       /{hostname}/wheels_driver_node/emergency_stop (:obj:`BoolStamped`): Emergency stop. Can stop the actual execution of
           the wheel commands by the motors if set to `True`. Set to `False` for nominal
           operations.
    Publishers:
       /{hostname}/wheels_driver_node/wheels_cmd_executed (:obj:`WheelsCmdStamped`): Publishes the actual commands executed,
           i.e. when the emergency flag is `False` it publishes the requested command, and
           when it is `True`: zero values for both motors.
        /{hostname}/left_wheel_encoder_node/tick (:obj:`WheelEncoderStamped`):
            Cumulative tick count on the left wheel.
        /{hostname}/right_wheel_encoder_node/tick (:obj:`WheelEncoderStamped`):
            Cumulative tick count on the right wheel.
    """

    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(MockWheelsDriverNode, self).__init__(node_name=node_name, node_type=NodeType.DRIVER)

        self.estop = False
        self.hostname = os.environ.get("VEHICLE_NAME")
        self.TICKS_PER_VEL = 10

        # Initialize the executed commands message
        self.msg_wheels_cmd = WheelsCmdStamped()

        # Publisher for wheels command wih execution time
        self.pub_wheels_cmd = rospy.Publisher(
            f"/{self.hostname}/wheels_driver_node/wheels_cmd_executed",
            WheelsCmdStamped, queue_size=1, dt_topic_type=TopicType.DRIVER
        )

        self.wheels = {}
        for name in ["left", "right"]:
            self.wheels[name] = {
                "pub_encoder_tick": rospy.Publisher(
                    f"/{self.hostname}/{name}_wheel_encoder_node/tick",
                    WheelEncoderStamped,
                    queue_size=1
                ),
                "ticks": 0,
                "vel": 0
            }

        # Subscribers
        self.sub_topic = rospy.Subscriber(
            f'/{self.hostname}/wheels_driver_node/wheels_cmd',
            WheelsCmdStamped,
            self.wheels_cmd_cb,
            queue_size=1
        )
        self.sub_e_stop = rospy.Subscriber(
            f"/{self.hostname}/wheels_driver_node//emergency_stop",
            BoolStamped,
            self.estop_cb,
            queue_size=1
        )

        self.log("Initialized.")

    def wheels_cmd_cb(self, msg):
        """
        Callback that sets wheels' speeds.

            Creates the wheels' speed message and publishes it. If the
            emergency stop flag is activated, publishes zero command.

            Args:
                msg (WheelsCmdStamped): velocity command
        """
        if self.estop:
            vel_left = 0.0
            vel_right = 0.0
        else:
            vel_left = msg.vel_left
            vel_right = msg.vel_right

        self.wheels["left"]["vel"] = vel_left
        self.wheels["left"]["vel"] = vel_left
        for name, wheel in self.wheels.items():
            if name == "left":
                wheel["vel"] = vel_left
            else:
                wheel["vel"] = vel_right

        # Put the wheel commands in a message and publish
        self.msg_wheels_cmd.header = msg.header
        # Record the time the command was given to the wheels_driver
        self.msg_wheels_cmd.header.stamp = rospy.get_rostime()
        self.msg_wheels_cmd.vel_left = vel_left
        self.msg_wheels_cmd.vel_right = vel_right
        self.pub_wheels_cmd.publish(self.msg_wheels_cmd)

    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            for _, wheel in self.wheels.items():
                wheel["ticks"] += self.TICKS_PER_VEL * wheel["vel"]
                message = WheelEncoderStamped()
                message.data = int(wheel["ticks"])
                message.resolution = 135
                message.type = 1
                wheel["pub_encoder_tick"].publish(message)
            rate.sleep()

    def estop_cb(self, msg):
        """
        Callback that enables/disables emergency stop

            Args:
                msg (BoolStamped): emergency_stop flag
        """

        self.estop = msg.data
        if self.estop:
            self.log("Emergency Stop Activated")
        else:
            self.log("Emergency Stop Released")

    def on_shutdown(self):
        """
        Shutdown procedure.

        Publishes a zero velocity command at shutdown.
        """
        for _, wheel in self.wheels.items():
            wheel["vel"] = 0


if __name__ == "__main__":
    # Initialize the node with rospy
    node = MockWheelsDriverNode(node_name="mock_wheels_driver_node")
    node.run()
    # Keep it spinning to keep the node alive
    rospy.spin()
