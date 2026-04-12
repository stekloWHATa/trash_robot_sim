#!/usr/bin/env python3
"""
map_check.py — сравнивает /map с реальными препятствиями из SDF.

Запуск (пока идёт симуляция):
    ros2 run trash_robot_sim map_check.py

Выводит каждые 5с:
  - сколько занятых ячеек совпадают с настоящими стенами
  - сколько ложные срабатывания (чёрная ячейка, стены нет)
  - координаты первых 30 ложных срабатываний
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np

# ── Настоящие препятствия из trash_world.sdf (center_x, center_y, half_x, half_y) ──
REAL_OBSTACLES = [
    ('wall_N',     0.0,  13.0,  13.0, 0.15),
    ('wall_S',     0.0, -13.0,  13.0, 0.15),
    ('wall_E',    13.0,   0.0,  0.15, 13.0),
    ('wall_W',   -13.0,   0.0,  0.15, 13.0),
    ('barrier_A',  1.0,   5.0,   4.0, 0.15),
    ('barrier_B', -6.0,   1.0,  0.15,  4.0),
    ('barrier_C',  3.0,   0.0,   3.0, 0.15),
    ('barrier_D',  5.0,  -4.0,  0.15,  3.0),
    ('barrier_E', -1.5,  -5.0,   3.5, 0.15),
]
# Допуск: ячейка считается «правильной» если её центр попадает в препятствие ± margin
MARGIN = 0.20   # метров (2 ячейки запаса)


def nearest_obstacle(wx, wy):
    best_name, best_d = '?', 1e9
    for name, cx, cy, hx, hy in REAL_OBSTACLES:
        d = ((wx - cx)**2 + (wy - cy)**2) ** 0.5
        if d < best_d:
            best_d, best_name = d, name
    return best_name, best_d


def in_any_obstacle(wx, wy, margin=MARGIN):
    for name, cx, cy, hx, hy in REAL_OBSTACLES:
        if abs(wx - cx) <= hx + margin and abs(wy - cy) <= hy + margin:
            return name
    return None


class MapChecker(Node):
    def __init__(self):
        super().__init__('map_checker')
        self._odom_x0 = 0.0
        self._odom_y0 = 2.0   # из лога: offset=(0.000, 2.000)

        self.create_subscription(OccupancyGrid, '/map', self._cb, 1)
        self.create_timer(5.0, self._report)
        self._last_msg = None
        self.get_logger().info('map_check: ждём /map...')

    def _cb(self, msg):
        self._last_msg = msg

    def _report(self):
        msg = self._last_msg
        if msg is None:
            self.get_logger().info('карта ещё не получена')
            return

        res = msg.info.resolution
        ox  = msg.info.origin.position.x   # odom frame
        oy  = msg.info.origin.position.y
        W   = msg.info.width
        H   = msg.info.height
        grid = np.array(msg.data, dtype=np.int8).reshape(H, W)

        rows, cols = np.where(grid == 100)
        if len(rows) == 0:
            self.get_logger().info('занятых ячеек нет')
            return

        # Преобразование: odom → world
        # odom_x = ox + col*res,  world_x = odom_x - odom_x0
        # odom_y = oy + row*res,  world_y = odom_y - odom_y0
        wx_arr = (ox + cols * res) - self._odom_x0
        wy_arr = (oy + rows * res) - self._odom_y0

        true_pos  = []
        false_pos = []
        for wx, wy in zip(wx_arr, wy_arr):
            name = in_any_obstacle(wx, wy)
            if name:
                true_pos.append((wx, wy, name))
            else:
                false_pos.append((wx, wy))

        print('\n' + '='*70)
        print(f'MAP CHECK  |  frame={msg.header.frame_id}  '
              f'origin=({ox:.2f},{oy:.2f})  res={res}m  {W}x{H}')
        print(f'  Занятых ячеек всего : {len(rows)}')
        print(f'  ✓ На реальном препятствии (+{MARGIN}м допуск): {len(true_pos)}')
        print(f'  ✗ ЛОЖНЫЕ СРАБАТЫВАНИЯ (стены нет): {len(false_pos)}')

        if false_pos:
            print()
            print(f'  Первые {min(30,len(false_pos))} ложных (world coords):')
            print(f'  {"wx":>8}  {"wy":>8}  ближайшее препятствие  расстояние')
            shown = sorted(false_pos, key=lambda p: nearest_obstacle(*p)[1])
            for wx, wy in shown[:30]:
                name, d = nearest_obstacle(wx, wy)
                print(f'  {wx:+8.2f}  {wy:+8.2f}  {name:<14}  {d:.2f} м')
            if len(false_pos) > 30:
                print(f'  ... и ещё {len(false_pos)-30} ложных')
        else:
            print('  → ложных срабатываний НЕТ, карта точная!')
        print('='*70)


def main():
    rclpy.init()
    node = MapChecker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
