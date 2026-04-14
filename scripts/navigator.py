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
import logging
import os

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Polygon, PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan

# ── Файловый лог: /tmp/trash_nav.log (перезаписывается при каждом старте) ── #
_LOG_PATH = '/tmp/trash_nav.log'
logging.basicConfig(
    filename=_LOG_PATH,
    filemode='w',
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
_flog = logging.getLogger('nav')


# ─── ПАРАМЕТРЫ КАРТЫ ─────────────────────────────────────────────────────── #
RESOLUTION   = 0.25
MAP_HALF     = 13.0
GRID_N       = int(2 * MAP_HALF / RESOLUTION)   # 104

# Робот: тело 1.2×0.8м, полудиагональ 0.72м.
# INFLATE_M = 1.50м (6 ячеек): жёсткий запрет — путь ≥1.50м от сырого препятствия.
# SOFT_R    = 2 ячейки (0.50м): мягкая зона — ячейки в 0–0.50м от жёсткой границы
#             стоят SOFT_K дороже. A* предпочтёт середину коридора, а не его край.
# Результат: путь проходит по центру коридора, с запасом от стен.
INFLATE_M    = 1.50
INFLATE_C    = max(1, int(INFLATE_M / RESOLUTION))   # 6 — для статической карты
# Инфляция скана МЕНЬШЕ статической: существующие стены надуты на 1.5м,
# поэтому лидарные хиты по ним не выйдут за статическую зону.
# Только новые объекты (мусор) добавят блокировку.
# 1.0м > полудиагональ робота 0.72м → достаточно для объезда.
SCAN_INFLATE = 4           # 4 ячейки = 1.0м — инфляция динамических хитов
SOFT_R       = 2          # ячейки мягкой зоны от жёсткой границы
SOFT_K       = 6.0        # штраф в стоимости движения через мягкую зону

# ─── ДВИЖЕНИЕ — значения по умолчанию (переопределяются из params.yaml) ───── #
# Используются только как fallback при отсутствии внешних параметров.
_DEF_MAX_LIN      = 0.65
_DEF_MAX_ANG      = 0.9
_DEF_LOOKAHEAD    = 1.5
_DEF_GOAL_R       = 0.90
_DEF_WAYPOINT_R   = 0.50
_DEF_GOAL_TIMEOUT = 120.0
_DEF_ROW_SPACING  = 2.5
_DEF_AREA_MARGIN  = 0.8


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
    """Прямоугольная (L∞) инфляция: блокирует все ячейки в квадрате r×r."""
    out = grid.copy()
    ys, xs = np.where(grid)
    for cy, cx in zip(ys, xs):
        y0 = max(0, cy - r)
        y1 = min(GRID_N - 1, cy + r)
        x0 = max(0, cx - r)
        x1 = min(GRID_N - 1, cx + r)
        out[y0:y1 + 1, x0:x1 + 1] = True
    return out


def make_cost_grid(blocked):
    """Строит карту стоимости для A*.
    Свободные ячейки вблизи жёсткой границы (SOFT_R ячеек = SOFT_R*RESOLUTION м)
    получают штраф SOFT_K, поэтому A* прокладывает путь по центру коридора,
    а не вплотную к стенам."""
    soft_zone = inflate(blocked, SOFT_R) & ~blocked
    cost = np.ones((GRID_N, GRID_N), dtype=np.float32)
    cost[soft_zone] = SOFT_K
    return cost


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


def astar(grid, start, goal, cost_grid=None):
    """A* с опциональной картой стоимости.
    cost_grid: float32 массив размера (GRID_N, GRID_N), стоимость прохода через ячейку.
    Без cost_grid — равномерная стоимость 1.0 (стандартный A*).
    С cost_grid — A* предпочитает ячейки с низкой стоимостью (центры коридоров)."""
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
        for dx, dy, move_cost in _MOVES:
            nx, ny = cx+dx, cy+dy
            if not (0 <= nx < GRID_N and 0 <= ny < GRID_N) or grid[ny, nx]:
                continue
            nb = (nx, ny)
            cell_cost = float(cost_grid[ny, nx]) if cost_grid is not None else 1.0
            ng = g[cur] + move_cost * cell_cost
            if ng < g.get(nb, 1e18):
                came[nb] = cur
                g[nb] = ng
                heapq.heappush(heap, (ng + _octile(nx, ny, gx, gy), nb))
    return None


# ─── УЗЕЛ ────────────────────────────────────────────────────────────────── #

class CoverageNavigator(Node):

    def __init__(self):
        super().__init__('coverage_navigator')

        # ── ROS параметры (загружаются из config/params.yaml) ──────────────── #
        self.declare_parameter('row_spacing',          _DEF_ROW_SPACING)
        self.declare_parameter('area_margin',          _DEF_AREA_MARGIN)
        self.declare_parameter('max_linear_velocity',  _DEF_MAX_LIN)
        self.declare_parameter('max_angular_velocity', _DEF_MAX_ANG)
        self.declare_parameter('goal_radius',          _DEF_GOAL_R)
        self.declare_parameter('goal_timeout',         _DEF_GOAL_TIMEOUT)
        self.declare_parameter('lookahead',            _DEF_LOOKAHEAD)
        self.declare_parameter('waypoint_radius',      _DEF_WAYPOINT_R)
        # Позиция центра тела робота в мировых координатах при спавне.
        # DiffDrive в Gazebo Harmonic стартует с odom=(0,0), а не с мировой позиции.
        # Смещение вычисляется из первого сообщения /odom и этих параметров.
        self.declare_parameter('spawn_x', 0.0)
        self.declare_parameter('spawn_y', -2.0)

        self.row_spacing   = self.get_parameter('row_spacing').value
        self.area_margin   = self.get_parameter('area_margin').value
        self.max_lin       = self.get_parameter('max_linear_velocity').value
        self.max_ang       = self.get_parameter('max_angular_velocity').value
        self.goal_r        = self.get_parameter('goal_radius').value
        self.goal_timeout  = self.get_parameter('goal_timeout').value
        self.lookahead     = self.get_parameter('lookahead').value
        self.waypoint_r    = self.get_parameter('waypoint_radius').value
        self._spawn_x      = self.get_parameter('spawn_x').value
        self._spawn_y      = self.get_parameter('spawn_y').value

        self._grid      = np.zeros((GRID_N, GRID_N), dtype=bool)
        self._scan_grid = np.zeros((GRID_N, GRID_N), dtype=bool)
        self._last_scan_nav: LaserScan | None = None
        self._build_static_map()

        self._x = 0.0; self._y = 0.0; self._yaw = 0.0
        self._odom_ok = False
        # Смещение odom→world: world = odom - _odom_offset_*
        # Вычисляется однократно из первого пакета /odom
        self._odom_x0: float | None = None
        self._odom_y0: float | None = None

        # Список точек покрытия (генерируется из /scan_area)
        self._coverage_goals: list[tuple[float, float]] = []
        self._goal_idx  = 0
        self._path: list[tuple[float, float]] = []
        self._wp_idx    = 0
        self._state     = 'WAIT_AREA'
        self._t_start   = 0.0

        # Детектор застревания: если позиция не меняется дольше STUCK_TIMEOUT — сдать назад
        self._stuck_x   = 0.0
        self._stuck_y   = 0.0
        self._t_stuck   = time.time()
        self._backing   = False
        self._t_back    = 0.0
        STUCK_TIMEOUT   = 5.0    # с — время без движения до признания застревания
        STUCK_DIST      = 0.15   # м — минимальное смещение за STUCK_TIMEOUT
        BACK_DURATION   = 1.5    # с — время движения назад
        self._STUCK_TIMEOUT  = STUCK_TIMEOUT
        self._STUCK_DIST     = STUCK_DIST
        self._BACK_DURATION  = BACK_DURATION

        self._pub      = self.create_publisher(Twist, '/cmd_vel', 10)
        self._path_pub = self.create_publisher(Path,  '/nav_path', 10)
        self.create_subscription(Odometry,    '/odom',       self._odom_cb,      10)
        self.create_subscription(LaserScan,   '/scan',       self._scan_cb_nav,  10)
        self.create_subscription(Polygon,     '/scan_area',  self._area_cb,      10)
        self.create_subscription(PoseStamped, '/goal_pose',  self._goal_pose_cb, 10)
        self.create_timer(0.1, self._loop)
        self.create_timer(3.0, self._log_status)  # периодический дамп состояния
        _flog.info(f'Navigator started. Log: {_LOG_PATH}')

        self.get_logger().info(
            'Coverage Navigator готов. Жду область /scan_area.\n'
            'Пример:\n  ros2 topic pub /scan_area geometry_msgs/msg/Polygon '
            '"{points: [{x: -9.0, y: -9.0, z: 0.0}, {x: 9.0, y: 9.0, z: 0.0}]}" --once'
        )

    # ── Lidar: динамические препятствия ──────────────────────────────────── #

    def _scan_cb_nav(self, msg: LaserScan):
        """Строит сетку препятствий из текущего скана (обновляется каждые 100мс)."""
        self._last_scan_nav = msg
        if not self._odom_ok:
            return
        raw = np.zeros((GRID_N, GRID_N), dtype=bool)
        for i, r in enumerate(msg.ranges):
            if math.isinf(r) or math.isnan(r) or r < 0.85 or r > 8.0:
                continue
            angle = self._yaw + msg.angle_min + i * msg.angle_increment
            ex = self._x + r * math.cos(angle)
            ey = self._y + r * math.sin(angle)
            gx, gy = w2g(ex, ey)
            raw[gy, gx] = True
        self._scan_grid = inflate(raw, SCAN_INFLATE)

    def _next_wp_blocked(self) -> bool:
        """True если следующий waypoint попал в зону НОВОГО (динамического) препятствия.
        Проверяем только scan_grid & ~static_grid: статические стены/барьеры A* уже знает,
        срабатываем только на реально новые объекты на пути."""
        if self._wp_idx >= len(self._path):
            return False
        wx, wy = self._path[self._wp_idx]
        gx, gy = w2g(wx, wy)
        # Только новые препятствия (не покрытые статической картой)
        new_obs = bool(self._scan_grid[gy, gx] and not self._grid[gy, gx])
        if new_obs:
            _flog.warning(
                f'WAYPOINT BLOCKED (new obstacle): wp#{self._wp_idx} '
                f'world=({wx:.2f},{wy:.2f}) grid=({gx},{gy}) → replan'
            )
        return new_obs

    def _log_status(self):
        """Периодический дамп состояния робота в файл (каждые 3с)."""
        goal_str = 'none'
        if self._coverage_goals and self._goal_idx < len(self._coverage_goals):
            gx, gy = self._coverage_goals[self._goal_idx]
            goal_str = f'({gx:.1f},{gy:.1f})'
        dist_to_goal = 0.0
        if self._coverage_goals and self._goal_idx < len(self._coverage_goals):
            gx, gy = self._coverage_goals[self._goal_idx]
            dist_to_goal = math.hypot(gx - self._x, gy - self._y)
        path_wp = f'{self._wp_idx}/{len(self._path)}' if self._path else '—'
        scan_blocked = int(self._scan_grid.sum())
        _flog.info(
            f'STATUS | state={self._state} '
            f'pos=({self._x:.2f},{self._y:.2f}) yaw={math.degrees(self._yaw):.1f}° '
            f'goal={goal_str}[{self._goal_idx}/{len(self._coverage_goals)}] '
            f'dist={dist_to_goal:.2f}m '
            f'wp={path_wp} '
            f'scan_blocked={scan_blocked} '
            f'odom_ok={self._odom_ok} backing={self._backing}'
        )

    # ── Статическая карта ─────────────────────────────────────────────────── #

    def _build_static_map(self):
        raw = np.zeros((GRID_N, GRID_N), dtype=bool)
        walls = [
            # Только внешние стены (±13м).
            # Инфляция 6 ячеек (1.5м) блокирует путь вплотную к стенам.
            ( 0.0,  13.0,  13.0,  0.15),
            ( 0.0, -13.0,  13.0,  0.15),
            ( 13.0,  0.0,  0.15, 13.0),
            (-13.0,  0.0,  0.15, 13.0),
        ]
        for cx, cy, hx, hy in walls:
            gx0, gy0 = w2g(cx - hx, cy - hy)
            gx1, gy1 = w2g(cx + hx, cy + hy)
            raw[gy0:gy1+1, gx0:gx1+1] = True

        self._grid      = inflate(raw, INFLATE_C)
        self._cost_grid = make_cost_grid(self._grid)

    # ── Callback: клик в RViz ("2D Goal Pose") ───────────────────────────── #

    def _goal_pose_cb(self, msg: PoseStamped):
        # RViz публикует в Fixed Frame = 'odom'. Переводим в мировые координаты.
        ox = msg.pose.position.x
        oy = msg.pose.position.y
        x0 = self._odom_x0 if self._odom_x0 is not None else 0.0
        y0 = self._odom_y0 if self._odom_y0 is not None else 0.0
        gx = ox - x0
        gy = oy - y0
        self._coverage_goals = [(gx, gy)]
        self._goal_idx = 0
        self._done_logged = False
        self._state = 'PLANNING' if self._odom_ok else 'WAIT_ODOM'
        self.get_logger().info(f'[RViz] Цель odom=({ox:.1f},{oy:.1f}) → world=({gx:.1f},{gy:.1f})')

    # ── Callback: оператор задаёт область ─────────────────────────────────── #

    def _area_cb(self, msg: Polygon):
        if len(msg.points) < 2:
            self.get_logger().warn('scan_area: нужно минимум 2 точки!')
            return

        xs = [p.x for p in msg.points]
        ys = [p.y for p in msg.points]
        x_min = max(min(xs), -MAP_HALF + self.area_margin)
        x_max = min(max(xs),  MAP_HALF - self.area_margin)
        y_min = max(min(ys), -MAP_HALF + self.area_margin)
        y_max = min(max(ys),  MAP_HALF - self.area_margin)

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
        y = y_min + self.row_spacing / 2.0
        left_to_right = True
        while y <= y_max + self.row_spacing * 0.1:
            y_clamped = min(y, y_max)
            if left_to_right:
                goals.append((x_min, y_clamped))
                goals.append((x_max, y_clamped))
            else:
                goals.append((x_max, y_clamped))
                goals.append((x_min, y_clamped))
            y += self.row_spacing
            left_to_right = not left_to_right
        return goals

    # ── Callbacks ────────────────────────────────────────────────────────── #

    def _odom_cb(self, msg: Odometry):
        ox = msg.pose.pose.position.x
        oy = msg.pose.pose.position.y

        # При первом пакете вычисляем смещение odom→world.
        # DiffDrive в Gazebo Harmonic может стартовать как с (0,0), так и с мировой позицией.
        # Независимо от поведения: world = odom - (first_odom - spawn)
        if self._odom_x0 is None:
            self._odom_x0 = ox - self._spawn_x
            self._odom_y0 = oy - self._spawn_y
            info = (
                f'ODOM INIT: offset=({self._odom_x0:.3f},{self._odom_y0:.3f}) '
                f'spawn=({self._spawn_x},{self._spawn_y}) '
                f'first_odom=({ox:.3f},{oy:.3f})'
            )
            self.get_logger().info(info)
            _flog.info(info)

        self._x   = ox - self._odom_x0
        self._y   = oy - self._odom_y0
        q         = msg.pose.pose.orientation
        self._yaw = math.atan2(
            2.0*(q.w*q.z + q.x*q.y),
            1.0 - 2.0*(q.y*q.y + q.z*q.z),
        )
        self._odom_ok = True

    # ── Детектор застревания ─────────────────────────────────────────────── #

    def _check_stuck(self) -> bool:
        """Возвращает True если уже обрабатываем застревание (движение назад)."""
        now = time.time()

        # Активный откат назад
        if self._backing:
            if now - self._t_back < self._BACK_DURATION:
                cmd = Twist()
                cmd.linear.x = -0.25
                self._pub.publish(cmd)
                return True
            else:
                self._backing = False
                self._state = 'PLANNING'   # перепланировать после отката
                self._stop()
                return False

        # Проверяем, сдвинулся ли робот
        if math.hypot(self._x - self._stuck_x, self._y - self._stuck_y) > self._STUCK_DIST:
            self._stuck_x = self._x
            self._stuck_y = self._y
            self._t_stuck = now

        if now - self._t_stuck > self._STUCK_TIMEOUT and self._state == 'FOLLOWING':
            msg = (
                f'STUCK at world=({self._x:.2f},{self._y:.2f}) '
                f'yaw={math.degrees(self._yaw):.1f}° '
                f'goal_idx={self._goal_idx}/{len(self._coverage_goals)} '
                f'→ reversing'
            )
            self.get_logger().warn(msg)
            _flog.warning(msg)
            self._backing = True
            self._t_back  = now
            return True

        return False

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
                _flog.info(f'DONE: all {len(self._coverage_goals)} goals visited')
                self._state = 'DONE'
                return
            ok = self._plan()
            if ok:
                self._t_start = time.time()
                self._state = 'FOLLOWING'
                _flog.info(f'→ FOLLOWING goal #{self._goal_idx}')
            else:
                msg = f'SKIP goal #{self._goal_idx} {self._coverage_goals[self._goal_idx]}: no path'
                self.get_logger().warn(msg)
                _flog.warning(msg)
                self._goal_idx += 1

        elif self._state == 'FOLLOWING':
            # Перепланировать если следующий waypoint оказался заблокирован
            if self._next_wp_blocked():
                self._stop()
                self._state = 'PLANNING'
                return

            if self._check_stuck():
                return

            elapsed = time.time() - self._t_start
            if elapsed > self.goal_timeout:
                msg = (
                    f'TIMEOUT goal #{self._goal_idx} '
                    f'pos=({self._x:.2f},{self._y:.2f}) '
                    f'elapsed={elapsed:.1f}s'
                )
                self.get_logger().warn(msg)
                _flog.warning(msg)
                self._goal_idx += 1
                self._state = 'PLANNING'
                self._stop()
                return

            gx_w, gy_w = self._coverage_goals[self._goal_idx]
            dist = math.hypot(gx_w - self._x, gy_w - self._y)

            if dist < self.goal_r:
                _flog.info(
                    f'REACHED goal #{self._goal_idx} '
                    f'({gx_w:.1f},{gy_w:.1f}) '
                    f'dist={dist:.2f}m elapsed={elapsed:.1f}s'
                )
                self._goal_idx += 1
                self._state = 'PLANNING'
                self._stop()
                return

            if dist < self.lookahead:
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

        # Добавляем к статической карте только НОВЫЕ препятствия из лидара.
        # scan_grid содержит инфляцию всех видимых объектов (в т.ч. статических стен),
        # поэтому маскируем статическую карту: берём только то, что не покрыто self._grid.
        # Это предотвращает «переблокировку» A* от двойной инфляции барьеров/стен.
        new_obstacles = self._scan_grid & ~self._grid
        combined = self._grid | new_obstacles
        cost     = make_cost_grid(combined)

        scan_blocked = int(self._scan_grid.sum())
        new_blocked  = int(new_obstacles.sum())
        _flog.info(
            f'PLAN #{self._goal_idx}: '
            f'from world=({self._x:.2f},{self._y:.2f}) grid={start} '
            f'→ world=({gx_w:.2f},{gy_w:.2f}) grid={goal} '
            f'scan_blocked={scan_blocked} new_only={new_blocked} '
            f'start_in_combined={bool(combined[start[1], start[0]])} '
            f'goal_in_combined={bool(combined[goal[1], goal[0]])}'
        )

        path_grid = astar(combined, start, goal, cost)
        if path_grid is None:
            _flog.error(
                f'PLAN FAILED: no path '
                f'from {start} to {goal} '
                f'(start_blocked={bool(combined[start[1],start[0]])}, '
                f'goal_blocked={bool(combined[goal[1],goal[0]])})'
            )
            return False

        wpts = [g2w(gx, gy) for gx, gy in path_grid[::2]]
        wpts.append((gx_w, gy_w))
        self._path   = wpts
        self._wp_idx = 0
        _flog.info(f'PLAN OK: {len(path_grid)} grid cells → {len(wpts)} waypoints')
        self._publish_path(wpts)
        return True

    def _publish_path(self, wpts):
        """Публикует запланированный маршрут в /nav_path для отображения в RViz."""
        x0 = self._odom_x0 if self._odom_x0 is not None else 0.0
        y0 = self._odom_y0 if self._odom_y0 is not None else 0.0
        msg = Path()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        for wx, wy in wpts:
            ps = PoseStamped()
            ps.header = msg.header
            # Переводим world → odom для отображения в Fixed Frame
            ps.pose.position.x = wx + x0
            ps.pose.position.y = wy + y0
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self._path_pub.publish(msg)

    # ── Pure Pursuit ─────────────────────────────────────────────────────── #

    def _get_lookahead(self) -> tuple[float, float]:
        while self._wp_idx < len(self._path) - 1:
            wx, wy = self._path[self._wp_idx]
            if math.hypot(wx - self._x, wy - self._y) < self.waypoint_r:
                self._wp_idx += 1
            else:
                break

        cum = 0.0
        px, py = self._x, self._y
        for i in range(self._wp_idx, len(self._path)):
            wx, wy = self._path[i]
            seg = math.hypot(wx - px, wy - py)
            if cum + seg >= self.lookahead:
                t = (self.lookahead - cum) / seg if seg > 1e-6 else 0.0
                return (px + t*(wx-px), py + t*(wy-py))
            cum += seg
            px, py = wx, wy
        return self._path[-1]

    def _steer_to(self, tx, ty):
        dist = math.hypot(tx - self._x, ty - self._y)
        err  = adiff(math.atan2(ty - self._y, tx - self._x), self._yaw)
        cmd  = Twist()
        if abs(err) > math.radians(60):
            cmd.angular.z = float(np.clip(1.2 * math.copysign(1.0, err), -self.max_ang, self.max_ang))
        else:
            k = math.cos(err) ** 2
            cmd.linear.x  = float(np.clip(self.max_lin * k * min(1.0, dist/self.lookahead), 0.1, self.max_lin))
            cmd.angular.z = float(np.clip(1.4 * err, -self.max_ang, self.max_ang))
        self._pub.publish(cmd)

    def _direct_approach(self, tx, ty, dist):
        err = adiff(math.atan2(ty - self._y, tx - self._x), self._yaw)
        cmd = Twist()
        ang_lim = min(0.7, self.max_ang)
        if abs(err) > math.radians(80):
            cmd.angular.z = float(np.clip(0.65 * math.copysign(1.0, err), -ang_lim, ang_lim))
        else:
            cmd.linear.x  = float(np.clip(0.45 * dist, 0.08, 0.4))
            cmd.angular.z = float(np.clip(0.9 * err,   -ang_lim, ang_lim))
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
