#!/usr/bin/env python3
"""
map_builder.py — occupancy-grid карта из лидара + детекция мусора по цвету.

Публикует:
  /map            nav_msgs/OccupancyGrid   (100ms, transient local)
  /trash_markers  visualization_msgs/MarkerArray
  /trash_report   std_msgs/String          (каждые 5с)

Подписывается на:
  /odom               — поза робота
  /scan               — данные лидара
  /rgbd/image         — цветное изображение RGBD-камеры (детекция мусора)
  /rgbd/depth_image   — карта глубины RGBD-камеры (точное расстояние до объекта)
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan, Image
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, String


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

# ── Камера ────────────────────────────────────────────────────────────────── #
# horizontal_fov RGBD-камеры из SDF (используется для детекции вместо nav_camera)
RGBD_CAM_FOV = 1.5   # rad

# ── Классы мусора: (name, (r_min,r_max, g_min,g_max, b_min,b_max), color_rgb) #
TRASH_CLASSES = [
    ('plastic_bottle_green',  ( 0,  80, 120, 255,  0,  80), (0.0, 0.8, 0.0)),
    ('plastic_bottle_blue',   ( 0,  80,   0, 100,150, 255), (0.0, 0.3, 1.0)),
    ('can_red',               (140,255,   0,  60,  0,  60), (1.0, 0.0, 0.0)),
    ('can_silver',            (170,255, 170, 255,170, 255), (0.8, 0.8, 0.8)),
    ('cardboard_box',         (100,200,  70, 150, 20,  80), (0.7, 0.5, 0.2)),
    ('plastic_bag',           (180,255, 180, 255,180, 255), (0.9, 0.9, 0.9)),
    ('paper',                 (210,255, 210, 255,190, 255), (0.95,0.95,0.8)),
    ('bottle_glass',          (  0, 60,  50, 120,  0,  60), (0.0, 0.5, 0.0)),
]

_DEF_DETECT_THRESH = 0.08    # доля пикселей ROI нужного цвета
_DEF_MERGE_DIST    = 2.5     # м — ближе этого = тот же объект


class MapBuilder(Node):

    def __init__(self):
        super().__init__('map_builder')

        # ── ROS параметры ──────────────────────────────────────────────────── #
        self.declare_parameter('detect_threshold', _DEF_DETECT_THRESH)
        self.declare_parameter('merge_distance',   _DEF_MERGE_DIST)
        self.declare_parameter('lidar_min_valid',  LIDAR_MIN_VALID)
        # Мировые координаты тела робота при спавне (для коррекции odom-смещения)
        self.declare_parameter('spawn_x', 0.0)
        self.declare_parameter('spawn_y', -2.0)

        self.detect_thresh = self.get_parameter('detect_threshold').value
        self.merge_dist    = self.get_parameter('merge_distance').value
        self._lidar_min    = self.get_parameter('lidar_min_valid').value
        self._spawn_x      = self.get_parameter('spawn_x').value
        self._spawn_y      = self.get_parameter('spawn_y').value

        self._log_odds = np.zeros((GRID_N, GRID_N), dtype=np.float32)
        self._x = 0.0; self._y = 0.0; self._yaw = 0.0
        self._odom_ok  = False
        self._last_scan: LaserScan | None = None
        # Смещение odom→world: вычисляется из первого пакета /odom
        self._odom_x0: float | None = None
        self._odom_y0: float | None = None

        # Последнее изображение глубины от RGBD-камеры
        self._depth_arr: np.ndarray | None = None

        self._trash: dict[int, tuple] = {}   # id → (wx, wy, class_name)
        self._trash_id_counter = 0

        map_qos = QoSProfile(depth=1,
                             reliability=QoSReliabilityPolicy.RELIABLE,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

        self._map_pub    = self.create_publisher(OccupancyGrid, '/map', map_qos)
        self._marker_pub = self.create_publisher(MarkerArray,   '/trash_markers', 10)
        self._report_pub = self.create_publisher(String,        '/trash_report',  10)

        self.create_subscription(Odometry,  '/odom',               self._odom_cb,  10)
        self.create_subscription(LaserScan, '/scan',               self._scan_cb,  10)
        # Детекция мусора идёт через RGBD-камеру (широкий угол + карта глубины)
        self.create_subscription(Image,     '/rgbd/image',         self._img_cb,   10)
        self.create_subscription(Image,     '/rgbd/depth_image',   self._depth_cb, 10)

        self.create_timer(0.10, self._publish_map)
        self.create_timer(5.0,  self._publish_report)

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
        self._odom_ok = True

    # ── LaserScan → log-odds ─────────────────────────────────────────────── #

    def _scan_cb(self, msg: LaserScan):
        if not self._odom_ok:
            return
        self._last_scan = msg

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
                # Останавливаемся перед уже уверенно занятой ячейкой:
                # луч из другой стороны стены не должен «стирать» стену.
                if self._log_odds[by, bx] > 2.0:
                    break
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

    # ── Детекция мусора ──────────────────────────────────────────────────── #

    def _img_cb(self, msg: Image):
        if not self._odom_ok:
            return
        if msg.encoding not in ('rgb8', 'bgr8', 'RGB8', 'BGR8'):
            return

        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        if msg.encoding in ('bgr8', 'BGR8'):
            arr = arr[:, :, ::-1]

        # Нижние 65% кадра: камера наклонена 50° вниз, объекты на полу
        # попадают в нижнюю часть изображения.
        roi_start = int(msg.height * 0.35)
        roi = arr[roi_start:, :, :]
        total = roi.shape[0] * roi.shape[1]

        for cls_name, (rmin,rmax,gmin,gmax,bmin,bmax), color in TRASH_CLASSES:
            mask = (
                (roi[:,:,0] >= rmin) & (roi[:,:,0] <= rmax) &
                (roi[:,:,1] >= gmin) & (roi[:,:,1] <= gmax) &
                (roi[:,:,2] >= bmin) & (roi[:,:,2] <= bmax)
            )
            ratio = mask.sum() / total
            if ratio >= self.detect_thresh:
                # Горизонтальный центроид маски → угол в кадре
                cols = np.where(mask.any(axis=0))[0]
                if len(cols) == 0:
                    break
                cx_pixel = float(cols.mean())
                # Вертикальный центроид → координата в полном изображении
                rows = np.where(mask.any(axis=1))[0]
                cy_pixel = roi_start + float(rows.mean()) if len(rows) > 0 else msg.height * 0.75

                angle_off = (cx_pixel - msg.width / 2.0) / msg.width * RGBD_CAM_FOV
                world_ang = self._yaw + angle_off

                # 1. Приоритет — карта глубины RGBD (субпиксельная точность)
                dist = self._depth_dist_at(cx_pixel, cy_pixel)

                # 2. Запасной вариант — ближайший луч лидара
                if dist is None or not (0.3 < dist < 8.0):
                    dist = self._lidar_dist_at(world_ang)

                # 3. Последний резерв — фиксированная дистанция
                if dist is None or not (0.3 < dist < 8.0):
                    dist = 1.5

                tx = self._x + dist * math.cos(world_ang)
                ty = self._y + dist * math.sin(world_ang)
                self._register_trash(cls_name, color, tx, ty)
                break

    def _lidar_dist_at(self, world_angle: float) -> float | None:
        """Ближайший лидарный луч в направлении world_angle."""
        if self._last_scan is None:
            return None
        msg = self._last_scan
        sensor_angle = world_angle - self._yaw
        while sensor_angle >  math.pi: sensor_angle -= 2 * math.pi
        while sensor_angle < -math.pi: sensor_angle += 2 * math.pi

        if sensor_angle < msg.angle_min or sensor_angle > msg.angle_max:
            return None

        idx = int((sensor_angle - msg.angle_min) / msg.angle_increment)
        readings = []
        for di in range(-5, 6):
            i = idx + di
            if 0 <= i < len(msg.ranges):
                r = msg.ranges[i]
                if not math.isinf(r) and not math.isnan(r) and self._lidar_min < r < msg.range_max:
                    readings.append(r)
        return min(readings) if readings else None

    # ── RGBD глубина ─────────────────────────────────────────────────────── #

    def _depth_cb(self, msg: Image):
        """Сохраняем последнюю карту глубины от RGBD-камеры."""
        if msg.encoding == '32FC1':
            self._depth_arr = np.frombuffer(
                msg.data, dtype=np.float32
            ).reshape(msg.height, msg.width).copy()
        elif msg.encoding in ('16UC1', '16UC'):
            arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
            self._depth_arr = arr.astype(np.float32) / 1000.0  # мм → м

    def _depth_dist_at(self, cx_pixel: float, cy_pixel: float) -> float | None:
        """Возвращает дистанцию (м) до пикселя (cx, cy) из карты глубины.
        Усредняет патч 11×11 вокруг точки для устойчивости."""
        if self._depth_arr is None:
            return None
        h, w = self._depth_arr.shape
        cx = int(np.clip(cx_pixel, 0, w - 1))
        cy = int(np.clip(cy_pixel, 0, h - 1))
        x0, x1 = max(0, cx - 5), min(w, cx + 6)
        y0, y1 = max(0, cy - 5), min(h, cy + 6)
        patch = self._depth_arr[y0:y1, x0:x1]
        valid = patch[np.isfinite(patch) & (patch > 0.1)]
        return float(np.median(valid)) if len(valid) > 0 else None

    def _register_trash(self, cls_name: str, color: tuple, wx: float, wy: float):
        for tid, (tx, ty, tc) in self._trash.items():
            if math.hypot(wx - tx, wy - ty) < self.merge_dist:
                return   # уже знаем

        tid = self._trash_id_counter
        self._trash_id_counter += 1
        self._trash[tid] = (wx, wy, cls_name)
        self.get_logger().info(f'[Trash] #{tid} {cls_name} @ ({wx:.1f}, {wy:.1f})')
        self._publish_markers()

    # ── Маркеры ──────────────────────────────────────────────────────────── #

    def _publish_markers(self):
        arr = MarkerArray()
        del_m = Marker(); del_m.action = Marker.DELETEALL
        arr.markers.append(del_m)

        for tid, (wx, wy, cls_name) in self._trash.items():
            color = (0.5, 0.5, 0.5)
            for c, _, col in TRASH_CLASSES:
                if c == cls_name:
                    color = col; break

            m = Marker()
            m.header.frame_id = 'odom'
            m.header.stamp    = self.get_clock().now().to_msg()
            m.ns = 'trash'; m.id = tid
            m.type = Marker.SPHERE; m.action = Marker.ADD
            m.pose.position.x = wx; m.pose.position.y = wy; m.pose.position.z = 0.5
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.35
            m.color.r, m.color.g, m.color.b = color; m.color.a = 0.9
            arr.markers.append(m)

            t = Marker()
            t.header = m.header; t.ns = 'trash_labels'; t.id = tid + 10000
            t.type = Marker.TEXT_VIEW_FACING; t.action = Marker.ADD
            t.pose.position.x = wx; t.pose.position.y = wy; t.pose.position.z = 0.9
            t.pose.orientation.w = 1.0; t.scale.z = 0.22
            t.color.r = t.color.g = t.color.b = 1.0; t.color.a = 1.0
            t.text = f'#{tid} {cls_name}'
            arr.markers.append(t)

        self._marker_pub.publish(arr)

    # ── Отчёт ────────────────────────────────────────────────────────────── #

    def _publish_report(self):
        if not self._trash:
            return
        lines = [f'=== МУСОР: {len(self._trash)} объектов ===']
        counts: dict[str, int] = {}
        for tid, (wx, wy, cls) in sorted(self._trash.items()):
            lines.append(f'  #{tid:<3d} {cls:<28s} x={wx:+6.1f}  y={wy:+6.1f}')
            counts[cls] = counts.get(cls, 0) + 1
        lines.append('--- По типам ---')
        for cls, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            lines.append(f'  {cls:<28s}: {cnt}')
        msg = String(); msg.data = '\n'.join(lines)
        self._report_pub.publish(msg)


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
