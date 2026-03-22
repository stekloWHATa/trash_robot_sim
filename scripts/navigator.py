#!/usr/bin/env python3
"""
Coverage Navigator для trash_robot_sim.

Оператор публикует прямоугольную область в /scan_area
(geometry_msgs/Polygon, первые 2 точки = противоположные углы).
Робот строит маршрут зигзагом (boustrophedon) и объезжает его,
пока map_builder собирает карту и детектирует мусор.

Пример команды оператора (сканировать всю арену):
  ros2 topic pub /scan_area geometry_msgs/msg/Polygon \
    "{points: [{x: -9.0, y: -9.0, z: 0.0}, {x: 9.0, y: 9.0, z: 0.0}]}" --once
"""

import math
import heapq
import time

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Polygon, PoseStamped, Twist
from nav_msgs.msg import Odometry


# ─── ПАРАМЕТРЫ КАРТЫ ─────────────────────────────────────────────────────── #
RESOLUTION   = 0.25
MAP_HALF     = 10.5
GRID_N       = int(2 * MAP_HALF / RESOLUTION)   # 84

INFLATE_M    = 0.75
INFLATE_C    = max(1, int(INFLATE_M / RESOLUTION))   # 3

# ─── ДВИЖЕНИЕ ────────────────────────────────────────────────────────────── #
MAX_LIN      = 0.65
MAX_ANG      = 0.9
LOOKAHEAD    = 1.5
GOAL_R       = 0.90
WAYPOINT_R   = 0.50
GOAL_TIMEOUT = 120.0

# ─── ПОКРЫТИЕ ────────────────────────────────────────────────────────────── #
# Расстояние между строками зигзага — примерно ширина обзора камеры
ROW_SPACING  = 2.5     # м

# Отступ от краёв области чтобы не врезаться в стены
AREA_MARGIN  = 0.8     # м


# ─── УТИЛИТЫ ─────────────────────────────────────────────────────────────── #

def w2g(wx, wy):
    return (
        max(0, min(GRID_N - 1, int((wx + MAP_HALF) / RESOLUTION))),
        max(0, min(GRID_N - 1, int((wy + MAP_HALF) / RESOLUTION))),
    )


def g2w(gx, gy):
    return (
        gx * RESOLUTION - MAP_HALF + RESOLUTION * 0.5,
        gy * RESOLUTION - MAP_HALF + RESOLUTION * 0.5,
    )


def adiff(a, b):
    d = a - b
    while d >  math.pi: d -= 2 * math.pi
    while d < -math.pi: d += 2 * math.pi
    return d


def inflate(grid, r):
    out = grid.copy()
    ys, xs = np.where(grid)
    for cy, cx in zip(ys, xs):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx*dx + dy*dy <= r*r:
                    ny, nx = cy+dy, cx+dx
                    if 0 <= ny < GRID_N and 0 <= nx < GRID_N:
                        out[ny, nx] = True
    return out


# ─── A* ─────────────────────────────────────────────────────────────────── #

_MOVES = [
    (-1,-1,1.4142),(-1,0,1.0),(-1,1,1.4142),
    ( 0,-1,1.0),              ( 0,1,1.0),
    ( 1,-1,1.4142),( 1,0,1.0),( 1,1,1.4142),
]


def _octile(ax, ay, bx, by):
    dx, dy = abs(ax-bx), abs(ay-by)
    return max(dx, dy) + 0.4142 * min(dx, dy)


def _nearest_free(grid, cx, cy):
    if not grid[cy, cx]:
        return cx, cy
    for r in range(1, INFLATE_C + 4):
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                nx, ny = cx+dx, cy+dy
                if 0 <= nx < GRID_N and 0 <= ny < GRID_N and not grid[ny, nx]:
                    return nx, ny
    return cx, cy


def astar(grid, start, goal):
    sx, sy = _nearest_free(grid, start[0], start[1])
    gx, gy = _nearest_free(grid, goal[0],  goal[1])
    if (sx, sy) == (gx, gy):
        return [(sx, sy)]

    heap = [(0.0, (sx, sy))]
    came = {}
    g    = {(sx, sy): 0.0}

    while heap:
        _, cur = heapq.heappop(heap)
        cx, cy = cur
        if (cx, cy) == (gx, gy):
            path = []
            node = (gx, gy)
            while node in came:
                path.append(node)
                node = came[node]
            path.append((sx, sy))
            path.reverse()
            return path
        for dx, dy, cost in _MOVES:
            nx, ny = cx+dx, cy+dy
            if not (0 <= nx < GRID_N and 0 <= ny < GRID_N) or grid[ny, nx]:
                continue
            nb = (nx, ny)
            ng = g[cur] + cost
            if ng < g.get(nb, 1e18):
                came[nb] = cur
                g[nb] = ng
                heapq.heappush(heap, (ng + _octile(nx, ny, gx, gy), nb))
    return None


# ─── УЗЕЛ ────────────────────────────────────────────────────────────────── #

class CoverageNavigator(Node):

    def __init__(self):
        super().__init__('coverage_navigator')

        self._grid = np.zeros((GRID_N, GRID_N), dtype=bool)
        self._build_static_map()

        self._x = 0.0; self._y = 0.0; self._yaw = 0.0
        self._odom_ok = False

        # Список точек покрытия (генерируется из /scan_area)
        self._coverage_goals: list[tuple[float, float]] = []
        self._goal_idx  = 0
        self._path: list[tuple[float, float]] = []
        self._wp_idx    = 0
        self._state     = 'WAIT_AREA'
        self._t_start   = 0.0

        self._pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Odometry,    '/odom',       self._odom_cb,      10)
        self.create_subscription(Polygon,     '/scan_area',  self._area_cb,      10)
        self.create_subscription(PoseStamped, '/goal_pose',  self._goal_pose_cb, 10)
        self.create_timer(0.1, self._loop)

        self.get_logger().info(
            'Coverage Navigator готов. Жду область /scan_area.\n'
            'Пример:\n  ros2 topic pub /scan_area geometry_msgs/msg/Polygon '
            '"{points: [{x: -9.0, y: -9.0, z: 0.0}, {x: 9.0, y: 9.0, z: 0.0}]}" --once'
        )

    # ── Статическая карта ─────────────────────────────────────────────────── #

    def _build_static_map(self):
        raw = np.zeros((GRID_N, GRID_N), dtype=bool)
        walls = [
            # Внешние стены (±10м)
            ( 0.0,  10.0,  10.0,  0.15),
            ( 0.0, -10.0,  10.0,  0.15),
            ( 10.0,  0.0,  0.15, 10.0),
            (-10.0,  0.0,  0.15, 10.0),
            # barrier_A горизонт. y=5,  x от -3 до 5
            ( 1.0,  5.0,   4.0,  0.15),
            # barrier_B верт.    x=-6, y от -3 до 5
            (-6.0,  1.0,   0.15,  4.0),
            # barrier_C горизонт. y=0,  x от 0 до 6
            ( 3.0,  0.0,   3.0,  0.15),
            # barrier_D верт.    x=5,  y от -7 до -1
            ( 5.0, -4.0,   0.15,  3.0),
            # barrier_E горизонт. y=-5, x от -5 до 2
            (-1.5, -5.0,   3.5,  0.15),
        ]
        for cx, cy, hx, hy in walls:
            gx0, gy0 = w2g(cx - hx, cy - hy)
            gx1, gy1 = w2g(cx + hx, cy + hy)
            raw[gy0:gy1+1, gx0:gx1+1] = True

        self._grid = inflate(raw, INFLATE_C)

    # ── Callback: клик в RViz ("2D Goal Pose") ───────────────────────────── #

    def _goal_pose_cb(self, msg: PoseStamped):
        gx = msg.pose.position.x
        gy = msg.pose.position.y
        self._coverage_goals = [(gx, gy)]
        self._goal_idx = 0
        self._done_logged = False
        self._state = 'PLANNING' if self._odom_ok else 'WAIT_ODOM'
        self.get_logger().info(f'[RViz] Цель: ({gx:.1f}, {gy:.1f})')

    # ── Callback: оператор задаёт область ─────────────────────────────────── #

    def _area_cb(self, msg: Polygon):
        if len(msg.points) < 2:
            self.get_logger().warn('scan_area: нужно минимум 2 точки!')
            return

        xs = [p.x for p in msg.points]
        ys = [p.y for p in msg.points]
        x_min = max(min(xs), -MAP_HALF + AREA_MARGIN)
        x_max = min(max(xs),  MAP_HALF - AREA_MARGIN)
        y_min = max(min(ys), -MAP_HALF + AREA_MARGIN)
        y_max = min(max(ys),  MAP_HALF - AREA_MARGIN)

        if x_max - x_min < 1.0 or y_max - y_min < 1.0:
            self.get_logger().warn('scan_area: слишком маленькая область!')
            return

        self._coverage_goals = self._boustrophedon(x_min, y_min, x_max, y_max)
        self._goal_idx  = 0
        self._state     = 'PLANNING' if self._odom_ok else 'WAIT_ODOM'

        self.get_logger().info(
            f'Область: X[{x_min:.1f}..{x_max:.1f}] Y[{y_min:.1f}..{y_max:.1f}] '
            f'→ {len(self._coverage_goals)} точек покрытия'
        )

    def _boustrophedon(self, x_min, y_min, x_max, y_max):
        """Зигзаг-маршрут по области."""
        goals = []
        y = y_min + ROW_SPACING / 2.0
        left_to_right = True
        while y <= y_max + ROW_SPACING * 0.1:
            y_clamped = min(y, y_max)
            if left_to_right:
                goals.append((x_min, y_clamped))
                goals.append((x_max, y_clamped))
            else:
                goals.append((x_max, y_clamped))
                goals.append((x_min, y_clamped))
            y += ROW_SPACING
            left_to_right = not left_to_right
        return goals

    # ── Callbacks ────────────────────────────────────────────────────────── #

    def _odom_cb(self, msg: Odometry):
        self._x   = msg.pose.pose.position.x
        self._y   = msg.pose.pose.position.y
        q         = msg.pose.pose.orientation
        self._yaw = math.atan2(
            2.0*(q.w*q.z + q.x*q.y),
            1.0 - 2.0*(q.y*q.y + q.z*q.z),
        )
        self._odom_ok = True

    # ── Главный цикл ─────────────────────────────────────────────────────── #

    def _loop(self):
        if self._state == 'WAIT_AREA':
            pass   # ждём /scan_area

        elif self._state == 'WAIT_ODOM':
            if self._odom_ok:
                self._state = 'PLANNING'

        elif self._state == 'PLANNING':
            self._stop()
            if self._goal_idx >= len(self._coverage_goals):
                self._state = 'DONE'
                return
            ok = self._plan()
            if ok:
                self._t_start = time.time()
                self._state = 'FOLLOWING'
            else:
                self.get_logger().warn(
                    f'Точка {self._coverage_goals[self._goal_idx]} недостижима, пропускаю.'
                )
                self._goal_idx += 1

        elif self._state == 'FOLLOWING':
            if time.time() - self._t_start > GOAL_TIMEOUT:
                self.get_logger().warn(f'Таймаут точки {self._goal_idx}, пропускаю.')
                self._goal_idx += 1
                self._state = 'PLANNING'
                self._stop()
                return

            gx_w, gy_w = self._coverage_goals[self._goal_idx]
            dist = math.hypot(gx_w - self._x, gy_w - self._y)

            if dist < GOAL_R:
                self._goal_idx += 1
                self._state = 'PLANNING'
                self._stop()
                return

            if dist < LOOKAHEAD:
                self._direct_approach(gx_w, gy_w, dist)
            else:
                lx, ly = self._get_lookahead()
                self._steer_to(lx, ly)

        elif self._state == 'DONE':
            self._stop()
            if not getattr(self, '_done_logged', False):
                n = len(self._coverage_goals)
                self.get_logger().info(
                    f'Сканирование завершено! Объехал {n} точек покрытия. '
                    f'Смотри /trash_markers и /trash_report.'
                )
                self._done_logged = True

    # ── Планировщик ──────────────────────────────────────────────────────── #

    def _plan(self) -> bool:
        gx_w, gy_w = self._coverage_goals[self._goal_idx]
        start = w2g(self._x, self._y)
        goal  = w2g(gx_w, gy_w)

        path_grid = astar(self._grid, start, goal)
        if path_grid is None:
            return False

        wpts = [g2w(gx, gy) for gx, gy in path_grid[::2]]
        wpts.append((gx_w, gy_w))
        self._path   = wpts
        self._wp_idx = 0
        return True

    # ── Pure Pursuit ─────────────────────────────────────────────────────── #

    def _get_lookahead(self) -> tuple[float, float]:
        while self._wp_idx < len(self._path) - 1:
            wx, wy = self._path[self._wp_idx]
            if math.hypot(wx - self._x, wy - self._y) < WAYPOINT_R:
                self._wp_idx += 1
            else:
                break

        cum = 0.0
        px, py = self._x, self._y
        for i in range(self._wp_idx, len(self._path)):
            wx, wy = self._path[i]
            seg = math.hypot(wx - px, wy - py)
            if cum + seg >= LOOKAHEAD:
                t = (LOOKAHEAD - cum) / seg if seg > 1e-6 else 0.0
                return (px + t*(wx-px), py + t*(wy-py))
            cum += seg
            px, py = wx, wy
        return self._path[-1]

    def _steer_to(self, tx, ty):
        dist = math.hypot(tx - self._x, ty - self._y)
        err  = adiff(math.atan2(ty - self._y, tx - self._x), self._yaw)
        cmd  = Twist()
        if abs(err) > math.radians(60):
            cmd.angular.z = float(np.clip(1.2 * math.copysign(1.0, err), -MAX_ANG, MAX_ANG))
        else:
            k = math.cos(err) ** 2
            cmd.linear.x  = float(np.clip(MAX_LIN * k * min(1.0, dist/LOOKAHEAD), 0.1, MAX_LIN))
            cmd.angular.z = float(np.clip(1.4 * err, -MAX_ANG, MAX_ANG))
        self._pub.publish(cmd)

    def _direct_approach(self, tx, ty, dist):
        err = adiff(math.atan2(ty - self._y, tx - self._x), self._yaw)
        cmd = Twist()
        if abs(err) > math.radians(80):
            cmd.angular.z = float(np.clip(0.65 * math.copysign(1.0, err), -0.7, 0.7))
        else:
            cmd.linear.x  = float(np.clip(0.45 * dist, 0.08, 0.4))
            cmd.angular.z = float(np.clip(0.9 * err,   -0.7, 0.7))
        self._pub.publish(cmd)

    def _stop(self):
        self._pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = CoverageNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
