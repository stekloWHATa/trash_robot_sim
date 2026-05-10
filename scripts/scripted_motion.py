#!/usr/bin/env python3
"""Waypoint motion for the detection demo video."""

import math
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


DEFAULT_WAYPOINTS = [
    0.00, -2.00,
    1.80, -1.40,
    3.60, -3.00,
    5.40, -1.40,
    7.20, -3.00,
    9.00, -1.40,
    10.80, -3.00,
]


def _yaw_from_quat(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _wrap_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class ScriptedMotion(Node):
    def __init__(self):
        super().__init__('scripted_motion')
        self.declare_parameter('cmd_topic', '/cmd_vel')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('loop', False)
        self.declare_parameter('spawn_x', 0.0)
        self.declare_parameter('spawn_y', -2.0)
        self.declare_parameter('waypoints', DEFAULT_WAYPOINTS)
        self.declare_parameter('waypoint_radius', 0.25)
        self.declare_parameter('hold_seconds', 1.0)
        self.declare_parameter('linear_speed', 0.42)
        self.declare_parameter('angular_gain', 1.8)
        self.declare_parameter('max_angular_speed', 0.85)

        self._pub = self.create_publisher(
            Twist, self.get_parameter('cmd_topic').value, 10)
        self.create_subscription(
            Odometry, self.get_parameter('odom_topic').value, self._odom_cb, 10)

        self._loop = bool(self.get_parameter('loop').value)
        self._spawn_x = float(self.get_parameter('spawn_x').value)
        self._spawn_y = float(self.get_parameter('spawn_y').value)
        self._waypoints = self._parse_waypoints(
            self.get_parameter('waypoints').value)
        self._waypoint_radius = float(self.get_parameter('waypoint_radius').value)
        self._hold_seconds = float(self.get_parameter('hold_seconds').value)
        self._linear_speed = float(self.get_parameter('linear_speed').value)
        self._angular_gain = float(self.get_parameter('angular_gain').value)
        self._max_angular_speed = float(self.get_parameter('max_angular_speed').value)

        self._odom_x0 = None
        self._odom_y0 = None
        self._x = self._spawn_x
        self._y = self._spawn_y
        self._yaw = 0.0
        self._odom_ok = False
        self._wp_idx = 0
        self._hold_until = 0.0
        self._done = False
        self._timer = self.create_timer(0.1, self._tick)

        self.get_logger().info(
            f'Scripted zigzag motion started: {len(self._waypoints)} waypoints')

    def _parse_waypoints(self, values):
        flat = [float(value) for value in values]
        if len(flat) < 4 or len(flat) % 2:
            self.get_logger().warn(
                'Invalid scripted waypoints parameter, using defaults')
            flat = DEFAULT_WAYPOINTS
        return [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]

    def _odom_cb(self, msg):
        ox = msg.pose.pose.position.x
        oy = msg.pose.pose.position.y
        if self._odom_x0 is None:
            self._odom_x0 = ox - self._spawn_x
            self._odom_y0 = oy - self._spawn_y
            self.get_logger().info(
                f'Scripted odom offset=({self._odom_x0:.3f},{self._odom_y0:.3f})')

        self._x = ox - self._odom_x0
        self._y = oy - self._odom_y0
        self._yaw = _yaw_from_quat(msg.pose.pose.orientation)
        self._odom_ok = True

    def _tick(self):
        if self._done:
            return

        if not self._odom_ok:
            self._pub.publish(Twist())
            return

        now = time.time()
        if now < self._hold_until:
            self._pub.publish(Twist())
            return

        if self._wp_idx >= len(self._waypoints):
            if self._loop:
                self._wp_idx = 0
            else:
                self._pub.publish(Twist())
                self._done = True
                self.get_logger().info('Scripted zigzag motion finished')
                return

        tx, ty = self._waypoints[self._wp_idx]
        dx = tx - self._x
        dy = ty - self._y
        dist = math.hypot(dx, dy)
        if dist <= self._waypoint_radius:
            self.get_logger().info(
                f'Waypoint {self._wp_idx + 1}/{len(self._waypoints)} reached: '
                f'({tx:.2f}, {ty:.2f})')
            self._wp_idx += 1
            self._hold_until = now + self._hold_seconds
            self._pub.publish(Twist())
            return

        target_yaw = math.atan2(dy, dx)
        yaw_error = _wrap_angle(target_yaw - self._yaw)
        angular = max(
            -self._max_angular_speed,
            min(self._max_angular_speed, self._angular_gain * yaw_error),
        )

        cmd = Twist()
        if abs(yaw_error) < 1.15:
            heading_factor = max(0.20, math.cos(yaw_error))
            cmd.linear.x = min(self._linear_speed, 0.65 * dist) * heading_factor
        else:
            cmd.linear.x = 0.0
        cmd.angular.z = angular
        self._pub.publish(cmd)


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
