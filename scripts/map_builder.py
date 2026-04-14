#!/usr/bin/env python3
"""
map_builder.py — occupancy-grid карта из лидара.

Публикует:
  /map  nav_msgs/OccupancyGrid  (100ms, transient local)

Подписывается на:
  /odom  — поза робота (одометрия)
  /scan  — данные лидара (LaserScan)

Детекция мусора вынесена в отдельный нод detector.py (YOLOv8).
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan


# ── Параметры карты ───────────────────────────────────────────────────────── #
RESOLUTION  = 0.10
MAP_HALF    = 13.0
GRID_N      = int(2 * MAP_HALF / RESOLUTION)   # 260

L_OCC  =  0.85
L_FREE = -0.40
L_MIN  = -5.0
L_MAX  =  5.0

LIDAR_MAX_RANGE = 11.5
# Фильтр самодетекции: лидар видит стойку камеры на ~0.55м → игнорим < 0.70м
LIDAR_MIN_VALID = 0.70


class MapBuilder(Node):

    def __init__(self):
        super().__init__('map_builder')

        # ── ROS параметры ──────────────────────────────────────────────────── #
        self.declare_parameter('lidar_min_valid', LIDAR_MIN_VALID)
        self.declare_parameter('spawn_x', 0.0)
        self.declare_parameter('spawn_y', -2.0)

        self._lidar_min = self.get_parameter('lidar_min_valid').value
        self._spawn_x   = self.get_parameter('spawn_x').value
        self._spawn_y   = self.get_parameter('spawn_y').value

        self._log_odds = np.zeros((GRID_N, GRID_N), dtype=np.float32)
        self._x = 0.0; self._y = 0.0; self._yaw = 0.0
        self._angular_vel = 0.0
        self._odom_ok  = False
        self._last_scan: LaserScan | None = None
        self._odom_x0: float | None = None
        self._odom_y0: float | None = None

        map_qos = QoSProfile(depth=1,
                             reliability=QoSReliabilityPolicy.RELIABLE,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

        self._map_pub = self.create_publisher(OccupancyGrid, '/map', map_qos)

        self.create_subscription(Odometry,  '/odom', self._odom_cb, 10)
        self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)

        self.create_timer(0.10, self._publish_map)

        self.get_logger().info(
            f'MapBuilder: grid {GRID_N}x{GRID_N}, res {RESOLUTION}m'
        )

    # ── Утилиты ──────────────────────────────────────────────────────────── #

    def _w2g(self, wx, wy):
        return (
            max(0, min(GRID_N - 1, int((wx + MAP_HALF) / RESOLUTION))),
            max(0, min(GRID_N - 1, int((wy + MAP_HALF) / RESOLUTION))),
        )

    # ── Odometry ─────────────────────────────────────────────────────────── #

    def _odom_cb(self, msg: Odometry):
        ox = msg.pose.pose.position.x
        oy = msg.pose.pose.position.y

        if self._odom_x0 is None:
            self._odom_x0 = ox - self._spawn_x
            self._odom_y0 = oy - self._spawn_y
            self.get_logger().info(
                f'Odom offset: ({self._odom_x0:.3f}, {self._odom_y0:.3f})'
            )

        self._x = ox - self._odom_x0
        self._y = oy - self._odom_y0
        q = msg.pose.pose.orientation
        self._yaw = math.atan2(
            2.0*(q.w*q.z + q.x*q.y),
            1.0 - 2.0*(q.y*q.y + q.z*q.z),
        )
        self._angular_vel = msg.twist.twist.angular.z
        self._odom_ok = True

    # ── LaserScan → log-odds ─────────────────────────────────────────────── #

    # Порог угловой скорости для пропуска скана.
    # При повороте > 0.3 рад/с скан приходит с задержкой ~100мс относительно
    # odom → yaw уже изменился на ~1.7°–5.2°, хит приземляется на 3–5 ячеек
    # не туда → ложные занятые ячейки в свободном пространстве.
    _MAX_ANG_FOR_SCAN = 0.3   # рад/с

    def _scan_cb(self, msg: LaserScan):
        if not self._odom_ok:
            return
        self._last_scan = msg

        # Пропускаем обновление карты при активном повороте — угол в odom
        # успевает измениться на несколько градусов за время задержки скана,
        # что даёт ложные хиты в свободном пространстве.
        if abs(self._angular_vel) > self._MAX_ANG_FOR_SCAN:
            return

        ox, oy = self._w2g(self._x, self._y)

        for i, r in enumerate(msg.ranges):
            # Игнорируем: nan/inf, ниже мин. дальности, и самодетекцию (стойка камеры)
            if math.isinf(r) or math.isnan(r) or r < self._lidar_min:
                continue

            hit    = r < min(msg.range_max, LIDAR_MAX_RANGE)
            end_r  = r if hit else LIDAR_MAX_RANGE
            angle  = self._yaw + msg.angle_min + i * msg.angle_increment

            ex = self._x + end_r * math.cos(angle)
            ey = self._y + end_r * math.sin(angle)
            ex_g, ey_g = self._w2g(ex, ey)

            for bx, by in self._bresenham(ox, oy, ex_g, ey_g):
                # Не стираем уже подтверждённые стены: ячейка выше порога отображения
                # (log_odds > 1.5) считается постоянной — стены статичны, никто их не двигает.
                if self._log_odds[by, bx] <= 1.5:
                    self._log_odds[by, bx] = max(L_MIN, self._log_odds[by, bx] + L_FREE)

            if hit:
                self._log_odds[ey_g, ex_g] = min(L_MAX, self._log_odds[ey_g, ex_g] + L_OCC)

    @staticmethod
    def _bresenham(x0, y0, x1, y1):
        cells = []
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0
        while True:
            if x == x1 and y == y1:
                break
            cells.append((x, y))
            if len(cells) > 2000:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy; x += sx
            if e2 < dx:
                err += dx; y += sy
        return cells

    # ── Публикация карты ─────────────────────────────────────────────────── #

    def _publish_map(self):
        # Все ячейки по умолчанию белые (свободно = 0).
        # Только подтверждённые препятствия (2+ попадания) красятся чёрным.
        # Серые "неизведанные" зоны убраны: если лидар туда не дотягивался —
        # скорее всего там открытое пространство.
        occ = np.zeros((GRID_N, GRID_N), dtype=np.int8)
        occ[self._log_odds > 1.5] = 100

        # Карта строится в мировых координатах, но публикуется в фрейме 'odom'.
        # Origin карты смещаем на odom-offset, чтобы ячейки совпали с тем,
        # что видит RViz (Fixed Frame = odom).
        x0 = self._odom_x0 if self._odom_x0 is not None else 0.0
        y0 = self._odom_y0 if self._odom_y0 is not None else 0.0

        msg = OccupancyGrid()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.info.resolution        = RESOLUTION
        msg.info.width             = GRID_N
        msg.info.height            = GRID_N
        msg.info.origin.position.x = -MAP_HALF + x0
        msg.info.origin.position.y = -MAP_HALF + y0
        msg.info.origin.orientation.w = 1.0
        msg.data = occ.flatten().tolist()
        self._map_pub.publish(msg)



def main(args=None):
    rclpy.init(args=args)
    node = MapBuilder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
