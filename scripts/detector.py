#!/usr/bin/env python3
"""
detector.py — детекция и пространственная локализация бытового мусора через YOLOv8.

Архитектура:
  - YOLOv8n: нейросетевая детекция объектов на RGB-кадре RGBD-камеры
  - Карта глубины: определение дистанции до центра детектированного bbox
  - Пинхол-камера + RT-матрица: перевод пикселей → мировые координаты
  - База объектов с кластеризацией: объединяет повторные детекции одного предмета

Публикует:
  /trash_markers   visualization_msgs/MarkerArray  (сферы + подписи)
  /trash_report    std_msgs/String                 (каждые 5с)
  /detections_img  sensor_msgs/Image               (~4 Гц, bbox-визуализация)

Подписывается на:
  /rgbd/image         — RGB-кадр 640×480
  /rgbd/depth_image   — карта глубины 640×480 (float32, метры)
  /rgbd/camera_info   — матрица интринсик K
  /odom               — поза робота в мире

Параметры (ROS):
  model_path     — путь к файлу .pt (по умолчанию: share/models/yolov8n.pt)
  conf_thresh    — порог уверенности YOLO (default 0.35)
  merge_dist     — радиус слияния детекций, м (default 1.2)
  detect_rate    — частота запуска инференса, Гц (default 4)
"""

import math
import os
import threading
from datetime import datetime

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from ament_index_python.packages import get_package_share_directory

from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

try:
    from ultralytics import YOLO
    _YOLO_OK = True
except ImportError:
    _YOLO_OK = False


# ── Геометрия RGBD-камеры на роботе ──────────────────────────────────────── #
#
#  Тело робота (body link) — начало отсчёта позы из /odom.
#  Цепочка трансформаций SDF:
#    body  → desk_for_depth : t=(0.55, 0, 0.5),  R=I
#    desk  → RGBD_camera    : t=(0.05, 0, 0.33), R=Ry(+1.1 rad)
#  Суммарное смещение камеры от тела:
#    tx = 0.60 м (вперёд по оси X тела)
#    tz = 0.83 м (вверх, т.е. высота над полом ≈ 0.5+0.83=1.33 м)
#
#  Тангаж +1.1 рад (≈ 63°): в SDF Ry(+p) наклоняет ось X вниз
#  (Ry(p)*[1,0,0] = [cos(p), 0, -sin(p)] при правостороннем правиле).
#  Таким образом оптическая ось камеры направлена вперёд-вниз — смотрит на пол.
#
#  Конвенция SDF-камеры → ROS optical frame:
#    ROS optical Z  = SDF camera +X  (оптическая ось)
#    ROS optical X  = SDF camera -Y  (правая сторона кадра)
#    ROS optical Y  = SDF camera -Z  (вниз по кадру)

CAM_TX    = 0.60   # м, смещение вперёд от body center
CAM_TZ    = 0.83   # м, смещение вверх от body center (body centre at z=0.5)
CAM_PITCH = 1.1    # рад, тангаж SDF-ссылки камеры (нос вниз)

# ── Отображение COCO классов → категории мусора ──────────────────────────── #
#  Ключ — COCO class id (0-based), значение — человекочитаемое имя.
COCO_TRASH_MAP = {
    39: 'plastic_bottle',   # bottle
    40: 'glass_bottle',     # wine glass
    41: 'can_cup',          # cup  (Coke Can от YOLO часто = cup)
    43: 'knife',            # knife
    44: 'spoon',            # spoon
    45: 'bowl',             # bowl
    46: 'organic_waste',    # banana
    47: 'organic_waste',    # apple
    51: 'organic_waste',    # orange
    56: 'furniture',        # chair
    63: 'electronics',      # laptop
    64: 'electronics',      # mouse
    65: 'electronics',      # remote
    66: 'electronics',      # keyboard
    67: 'electronics',      # cell phone
    72: 'cardboard_paper',  # book
    73: 'misc_object',      # clock
    74: 'glass_bottle',     # vase
    75: 'sharp_object',     # scissors
    76: 'toy',              # teddy bear
    77: 'appliance',        # hair drier
    78: 'hygiene',          # toothbrush
}

# Цвета маркеров (r, g, b) для каждой категории
CATEGORY_COLOR = {
    'plastic_bottle':  (0.2, 0.6, 1.0),
    'glass_bottle':    (0.2, 0.8, 0.3),
    'can_cup':         (1.0, 0.2, 0.2),
    'cardboard_paper': (0.8, 0.6, 0.2),
    'organic_waste':   (0.6, 0.9, 0.2),
    'electronics':     (0.5, 0.5, 0.9),
    'sharp_object':    (1.0, 0.5, 0.0),
    'hygiene':         (0.9, 0.9, 0.9),
    'toy':             (1.0, 0.8, 0.0),
    'appliance':       (0.7, 0.4, 0.7),
    'furniture':       (0.6, 0.4, 0.2),
    'misc_object':     (0.8, 0.8, 0.5),
    'knife':           (0.9, 0.3, 0.1),
    'spoon':           (0.8, 0.8, 0.6),
    'bowl':            (0.6, 0.8, 0.8),
}


class Detector(Node):

    def __init__(self):
        super().__init__('trash_detector')

        # ── Директория для сохранения снимков ─────────────────────────── #
        self._save_dir = '/tmp/trash_detected'
        os.makedirs(self._save_dir, exist_ok=True)

        # ── Параметры ──────────────────────────────────────────────────── #
        pkg = get_package_share_directory('trash_robot_sim')
        default_model = os.path.join(pkg, 'models', 'yolov8n.pt')

        self.declare_parameter('model_path',  default_model)
        self.declare_parameter('conf_thresh', 0.35)
        self.declare_parameter('merge_dist',  1.2)
        self.declare_parameter('detect_rate', 4.0)
        self.declare_parameter('spawn_x',     0.0)
        self.declare_parameter('spawn_y',    -2.0)

        self._conf  = self.get_parameter('conf_thresh').value
        self._merge = self.get_parameter('merge_dist').value
        self._rate  = self.get_parameter('detect_rate').value
        self._spawn_x = self.get_parameter('spawn_x').value
        self._spawn_y = self.get_parameter('spawn_y').value

        # ── Загрузка YOLOv8 ────────────────────────────────────────────── #
        model_path = self.get_parameter('model_path').value
        self._model = None
        if _YOLO_OK:
            if os.path.isfile(model_path):
                self.get_logger().info(f'Загружаю YOLOv8: {model_path}')
                self._model = YOLO(model_path)
                self._model.fuse()
                self.get_logger().info('YOLOv8 готов')
            else:
                self.get_logger().warn(
                    f'Файл модели не найден: {model_path}\n'
                    f'  Запустите: yolo export model=yolov8n.pt format=pt\n'
                    f'  и скопируйте yolov8n.pt в {os.path.dirname(model_path)}'
                )
        else:
            self.get_logger().error(
                'ultralytics не установлен. pip install ultralytics'
            )

        # ── Состояние робота ───────────────────────────────────────────── #
        self._robot_x  = self._spawn_x
        self._robot_y  = self._spawn_y
        self._robot_yaw = 0.0
        self._odom_x0: float | None = None
        self._odom_y0: float | None = None
        self._odom_ok  = False

        # ── Интринсики камеры ──────────────────────────────────────────── #
        self._K: np.ndarray | None = None   # 3×3 матрица

        # ── Данные изображений ─────────────────────────────────────────── #
        self._img_lock   = threading.Lock()
        self._latest_rgb:   np.ndarray | None = None   # H×W×3 uint8
        self._latest_depth: np.ndarray | None = None   # H×W float32
        self._rgb_stamp  = None
        self._detections_pending = False

        # ── База обнаруженных объектов ─────────────────────────────────── #
        # id → {'x','y','category','label','conf','count'}
        self._trash: dict[int, dict] = {}
        self._trash_counter = 0

        # ── QoS ────────────────────────────────────────────────────────── #
        transient = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        # ── Публикаторы ────────────────────────────────────────────────── #
        self._pub_markers = self.create_publisher(
            MarkerArray, '/trash_markers', transient)
        self._pub_report  = self.create_publisher(
            String, '/trash_report', 10)
        self._pub_detimg  = self.create_publisher(
            Image, '/detections_img', 10)

        # ── Подписки ───────────────────────────────────────────────────── #
        self.create_subscription(Odometry,   '/odom',             self._odom_cb,    10)
        self.create_subscription(CameraInfo, '/rgbd/camera_info', self._caminfo_cb,  1)
        self.create_subscription(Image,      '/rgbd/image',       self._rgb_cb,     10)
        self.create_subscription(Image,      '/rgbd/depth_image', self._depth_cb,   10)

        # ── Таймеры ────────────────────────────────────────────────────── #
        detect_period = 1.0 / max(0.5, self._rate)
        self.create_timer(detect_period, self._detect_timer)
        self.create_timer(5.0,           self._report_timer)

        self.get_logger().info(
            f'Detector: conf={self._conf}, merge={self._merge}м, '
            f'rate={self._rate}Гц'
        )

    # ── Одометрия ────────────────────────────────────────────────────────── #

    def _odom_cb(self, msg: Odometry):
        ox = msg.pose.pose.position.x
        oy = msg.pose.pose.position.y
        if self._odom_x0 is None:
            self._odom_x0 = ox - self._spawn_x
            self._odom_y0 = oy - self._spawn_y
        self._robot_x = ox - self._odom_x0
        self._robot_y = oy - self._odom_y0
        q = msg.pose.pose.orientation
        self._robot_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        self._odom_ok = True

    # ── Интринсики камеры ─────────────────────────────────────────────────── #

    def _caminfo_cb(self, msg: CameraInfo):
        if self._K is None:
            self._K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.get_logger().info(
                f'Camera K: fx={self._K[0,0]:.1f} fy={self._K[1,1]:.1f} '
                f'cx={self._K[0,2]:.1f} cy={self._K[1,2]:.1f}'
            )

    # ── Приём изображений ─────────────────────────────────────────────────── #

    def _rgb_cb(self, msg: Image):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3)
        if msg.encoding in ('bgr8', 'BGR8'):
            arr = arr[:, :, ::-1]
        with self._img_lock:
            self._latest_rgb   = arr.copy()
            self._rgb_stamp    = msg.header.stamp

    def _depth_cb(self, msg: Image):
        if msg.encoding == '32FC1':
            arr = np.frombuffer(msg.data, dtype=np.float32).reshape(
                msg.height, msg.width).copy()
        elif msg.encoding in ('16UC1', '16UC'):
            raw = np.frombuffer(msg.data, dtype=np.uint16).reshape(
                msg.height, msg.width)
            arr = raw.astype(np.float32) / 1000.0
        else:
            return
        with self._img_lock:
            self._latest_depth = arr

    # ── Детекция ─────────────────────────────────────────────────────────── #

    def _detect_timer(self):
        """Периодический запуск YOLOv8 инференса."""
        if self._model is None or not self._odom_ok or self._K is None:
            return

        with self._img_lock:
            if self._latest_rgb is None or self._latest_depth is None:
                return
            rgb   = self._latest_rgb.copy()
            depth = self._latest_depth.copy()

        results = self._model(rgb, conf=self._conf, verbose=False)
        if not results:
            return

        det = results[0]
        annotated = det.plot()   # кадр с нарисованными bbox

        found_new = False
        for box in det.boxes:
            cls_id = int(box.cls[0].item())
            if cls_id not in COCO_TRASH_MAP:
                continue
            category = COCO_TRASH_MAP[cls_id]
            conf_val = float(box.conf[0].item())
            label    = det.names[cls_id]

            # Центр bbox в пикселях
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            u = (x1 + x2) / 2.0
            v = (y1 + y2) / 2.0

            # Глубина: медиана патча 11×11 вокруг центра bbox
            d = self._sample_depth(depth, u, v)
            if d is None:
                continue

            # Пространственная локализация: пиксель + глубина → мировые координаты
            wx, wy = self._pixel_to_world(u, v, d)
            if wx is None:
                continue

            if self._register(wx, wy, category, label, conf_val,
                              annotated, (u, v, x1, y1, x2, y2)):
                found_new = True

        if found_new:
            self._publish_markers()

        # Публикуем аннотированный кадр
        self._publish_det_img(annotated)

    def _sample_depth(self, depth: np.ndarray, u: float, v: float,
                      patch: int = 5) -> float | None:
        """Медиана глубины в патче patch×patch вокруг (u, v)."""
        h, w = depth.shape
        cu = int(np.clip(u, 0, w - 1))
        cv = int(np.clip(v, 0, h - 1))
        x0, x1 = max(0, cu - patch), min(w, cu + patch + 1)
        y0, y1 = max(0, cv - patch), min(h, cv + patch + 1)
        roi = depth[y0:y1, x0:x1]
        valid = roi[np.isfinite(roi) & (roi > 0.1) & (roi < 10.0)]
        if len(valid) < 3:
            return None
        return float(np.median(valid))

    def _pixel_to_world(self, u: float, v: float,
                        depth: float) -> tuple[float | None, float | None]:
        """
        Преобразование (u, v, depth) → (world_x, world_y).

        Цепочка трансформаций:
          1. Пиксель → 3D в ROS optical frame:
               X_opt = (u - cx) * Z / fx
               Y_opt = (v - cy) * Z / fy
               Z_opt = Z  (глубина)

          2. ROS optical → SDF camera link frame:
               X_sdf =  Z_opt    (оптическая ось = SDF +X)
               Y_sdf = -X_opt    (вправо opt  = влево SDF)
               Z_sdf = -Y_opt    (вниз opt    = вверх SDF)

          3. SDF camera → body: t=(0.60, 0, 0.83), Ry(CAM_PITCH)
               Ry(p)*[x,y,z] = [cos(p)*x+sin(p)*z, y, -sin(p)*x+cos(p)*z]

          4. Body → world: t=(rx, ry, 0.5), Rz(yaw)
        """
        if self._K is None:
            return None, None

        fx = self._K[0, 0]; fy = self._K[1, 1]
        cx = self._K[0, 2]; cy = self._K[1, 2]

        # Шаг 1: optical frame
        X_opt = (u - cx) * depth / fx
        Y_opt = (v - cy) * depth / fy
        Z_opt = depth

        # Шаг 2: SDF camera link frame
        X_sdf =  Z_opt
        Y_sdf = -X_opt
        Z_sdf = -Y_opt

        # Шаг 3: body frame
        p = CAM_PITCH
        cp, sp = math.cos(p), math.sin(p)
        X_body = cp * X_sdf + sp * Z_sdf + CAM_TX
        Y_body = Y_sdf
        Z_body = -sp * X_sdf + cp * Z_sdf + CAM_TZ

        # Шаг 4: world frame
        yaw = self._robot_yaw
        cy_r, sy_r = math.cos(yaw), math.sin(yaw)
        wx = self._robot_x + cy_r * X_body - sy_r * Y_body
        wy = self._robot_y + sy_r * X_body + cy_r * Y_body
        # wz = 0.5 + Z_body  # ≈ 0 для напольных объектов (контроль)

        return wx, wy

    # ── Регистрация объектов ──────────────────────────────────────────────── #

    def _register(self, wx: float, wy: float, category: str,
                  label: str, conf: float,
                  frame_bgr: np.ndarray | None = None,
                  bbox: tuple | None = None) -> bool:
        """
        Добавляет объект в базу или обновляет уверенность существующего.
        Возвращает True, если добавлен НОВЫЙ объект.
        При добавлении нового объекта сохраняет кроп bbox на диск.
        """
        # Слияние: ищем ближайший уже известный объект
        for obj in self._trash.values():
            if math.hypot(wx - obj['x'], wy - obj['y']) < self._merge:
                obj['x'] = 0.8 * obj['x'] + 0.2 * wx
                obj['y'] = 0.8 * obj['y'] + 0.2 * wy
                obj['count'] += 1
                if conf > obj['conf']:
                    obj['conf'] = conf
                return False

        # Новый объект
        tid = self._trash_counter
        self._trash_counter += 1
        self._trash[tid] = {
            'x': wx, 'y': wy,
            'category': category,
            'label': label,
            'conf': conf,
            'count': 1,
        }
        self.get_logger().info(
            f'[Trash] #{tid} {label}/{category} @ '
            f'({wx:.2f}, {wy:.2f})  conf={conf:.2f}'
        )

        # Сохраняем снимок нового объекта
        if frame_bgr is not None and bbox is not None:
            self._save_detection(tid, label, conf, frame_bgr, bbox)

        return True

    def _save_detection(self, tid: int, label: str, conf: float,
                        frame_bgr: np.ndarray, bbox: tuple):
        """Сохраняет аннотированный кадр и вырезанный bbox в /tmp/trash_detected/."""
        ts = datetime.now().strftime('%H%M%S_%f')[:10]
        safe_label = label.replace(' ', '_')

        # Полный кадр с bbox
        full_path = os.path.join(
            self._save_dir, f'{ts}_id{tid:03d}_{safe_label}_full.jpg')
        cv2.imwrite(full_path, frame_bgr)

        # Кроп bbox (с небольшим отступом)
        u, v, x1, y1, x2, y2 = bbox
        h, w = frame_bgr.shape[:2]
        pad = 20
        cx1 = max(0, int(x1) - pad)
        cy1 = max(0, int(y1) - pad)
        cx2 = min(w, int(x2) + pad)
        cy2 = min(h, int(y2) + pad)
        crop = frame_bgr[cy1:cy2, cx1:cx2]
        if crop.size > 0:
            crop_path = os.path.join(
                self._save_dir, f'{ts}_id{tid:03d}_{safe_label}_crop.jpg')
            cv2.imwrite(crop_path, crop)
        self.get_logger().info(f'  → сохранено: {full_path}')

    # ── Публикация маркеров ───────────────────────────────────────────────── #

    def _publish_markers(self):
        arr = MarkerArray()

        # Сначала удаляем все старые маркеры
        del_m = Marker()
        del_m.action = Marker.DELETEALL
        arr.markers.append(del_m)

        for tid, obj in self._trash.items():
            color = CATEGORY_COLOR.get(obj['category'], (0.7, 0.7, 0.7))

            # Сфера
            m = Marker()
            m.header.frame_id = 'odom'
            m.header.stamp    = self.get_clock().now().to_msg()
            m.ns = 'trash'; m.id = tid
            m.type   = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = obj['x']
            m.pose.position.y = obj['y']
            m.pose.position.z = 0.3
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.35
            m.color.r, m.color.g, m.color.b = color
            m.color.a = 0.9
            arr.markers.append(m)

            # Подпись
            t = Marker()
            t.header = m.header
            t.ns = 'trash_labels'; t.id = tid + 10000
            t.type   = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = obj['x']
            t.pose.position.y = obj['y']
            t.pose.position.z = 0.75
            t.pose.orientation.w = 1.0
            t.scale.z = 0.20
            t.color.r = t.color.g = t.color.b = 1.0
            t.color.a = 1.0
            t.text = (
                f'#{tid} {obj["label"]}\n'
                f'conf={obj["conf"]:.2f}  n={obj["count"]}'
            )
            arr.markers.append(t)

        self._pub_markers.publish(arr)

    # ── Публикация аннотированного кадра ─────────────────────────────────── #

    def _publish_det_img(self, bgr: np.ndarray):
        """Публикует BGR numpy-кадр как sensor_msgs/Image."""
        msg = Image()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'
        msg.height   = bgr.shape[0]
        msg.width    = bgr.shape[1]
        msg.encoding = 'bgr8'
        msg.step     = bgr.shape[1] * 3
        msg.data     = bgr.tobytes()
        self._pub_detimg.publish(msg)

    # ── Отчёт ────────────────────────────────────────────────────────────── #

    def _report_timer(self):
        if not self._trash:
            return
        lines = [f'=== МУСОР: {len(self._trash)} объектов ===']
        counts: dict[str, int] = {}
        for tid, obj in sorted(self._trash.items()):
            lines.append(
                f'  #{tid:<3d} {obj["label"]:<18s} ({obj["category"]:<18s}) '
                f'x={obj["x"]:+6.2f}  y={obj["y"]:+6.2f}  '
                f'conf={obj["conf"]:.2f}  n={obj["count"]}'
            )
            counts[obj['category']] = counts.get(obj['category'], 0) + 1
        lines.append('--- По категориям ---')
        for cat, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            lines.append(f'  {cat:<22s}: {cnt}')
        msg = String()
        msg.data = '\n'.join(lines)
        self._pub_report.publish(msg)
        self.get_logger().info(
            f'Детектировано объектов: {len(self._trash)}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = Detector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
