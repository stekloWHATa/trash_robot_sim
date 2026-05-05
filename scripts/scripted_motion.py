#!/usr/bin/env python3
"""Simple repeatable motion for the detection demo video."""

import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class ScriptedMotion(Node):
    def __init__(self):
        super().__init__('scripted_motion')
        self.declare_parameter('cmd_topic', '/cmd_vel')
        self.declare_parameter('loop', False)

        self._pub = self.create_publisher(
            Twist, self.get_parameter('cmd_topic').value, 10)
        self._loop = bool(self.get_parameter('loop').value)
        self._start = time.time()
        self._done = False
        self._timer = self.create_timer(0.1, self._tick)

        # duration, linear.x, angular.z. Tuned for a short video pass.
        self._phases = [
            (2.0, 0.00, 0.00),   # let Gazebo/RViz settle
            (5.0, 0.35, 0.00),   # approach trash cluster
            (3.0, 0.05, 0.35),   # pan right while still moving
            (3.0, 0.10, -0.35),  # pan left
            (2.0, 0.00, 0.00),   # stable final frame for markers
        ]

        self.get_logger().info('Scripted demo motion started')

    def _tick(self):
        if self._done:
            return

        elapsed = time.time() - self._start
        total = sum(p[0] for p in self._phases)
        if self._loop and total > 0.0:
            elapsed = elapsed % total

        acc = 0.0
        cmd = Twist()
        for duration, lin, ang in self._phases:
            acc += duration
            if elapsed <= acc:
                cmd.linear.x = lin
                cmd.angular.z = ang
                self._pub.publish(cmd)
                return

        self._pub.publish(Twist())
        self._done = True
        self.get_logger().info('Scripted demo motion finished')


def main(args=None):
    rclpy.init(args=args)
    node = ScriptedMotion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
