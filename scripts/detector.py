#!/usr/bin/env python3
"""
detector.py — детекция и пространственная локализация бытового мусора через YOLOv8.

Архитектура:
  - YOLOv8n: нейросетевая детекция объектов на RGB-кадре камеры
  - Карта глубины: определение дистанции до центра детектированного bbox
  - Пинхол-камера + RT-матрица: перевод пикселей → мировые координаты
  - База объектов с кластеризацией: объединяет повторные детекции одного предмета

Публикует:
  /trash_markers   visualization_msgs/MarkerArray  (сферы + подписи)
  /trash_report    std_msgs/String                 (каждые 5с)
  /detections_img  sensor_msgs/Image               (~4 Гц, bbox-визуализация)

Подписывается на:
  /rgbd/image/image          — RGB-кадр 640×480 (default camera_mode=rgbd)
  /rgbd/image/depth_image    — карта глубины 640×480 (float32, метры)
  /rgbd/image/camera_info    — матрица интринсик K
  /odom               — поза робота в мире

Параметры (ROS):
  model_path     — путь к файлу .pt (по умолчанию: share/models/yolov8s_trash.pt)
  conf_thresh    — порог уверенности YOLO (default 0.35)
  merge_dist     — радиус слияния детекций, м (default 1.2)
  detect_rate    — частота запуска инференса, Гц (default 4)
  camera_mode    — rgbd или nav_ground_plane
  log_path       — JSONL-журнал детекций для последующей оценки демо
"""

import json
import math
import os
import threading
import time
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


# ── Геометрия камер на роботе ─────────────────────────────────────────────── #
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

#  RGBD camera:
#    body -> desk_for_depth: t=(0.55, 0, 0.50)
#    desk -> RGBD_camera:   t=(0.05, 0, 0.33), Ry(+1.10)
#    итог: t=(0.60, 0, 0.83), horizontal_fov=1.5
#
#  NavCamera:
#    body -> nav_camera: t=(0.60, 0, 0.00), Ry(+0.45), horizontal_fov=1.0
#
#  Важно: RGB и depth должны быть из одной камеры. Поэтому default режим — rgbd.
BODY_Z = 0.50

CAMERA_PROFILES = {
    'rgbd': {
        'tx': 0.60,
        'tz': 0.83,
        'pitch': 1.10,
        'fov': 1.50,
        'rgb_topic': '/rgbd/image/image',
        'depth_topic': '/rgbd/image/depth_image',
        'camera_info_topic': '/rgbd/image/camera_info',
        'localization': 'depth',
        'frame_id': 'RGBD_camera',
    },
    'nav_ground_plane': {
        'tx': 0.60,
        'tz': 0.00,
        'pitch': 0.45,
        'fov': 1.00,
        'rgb_topic': '/camera/image',
        'depth_topic': '',
        'camera_info_topic': '',
        'localization': 'ground_plane',
        'frame_id': 'nav_camera',
    },
}

# ── Отображение COCO классов → категории мусора ──────────────────────────── #
#  Ключ — COCO class id (0-based), значение — человекочитаемое имя.
COCO_TRASH_MAP = {
    # ── Прямые попадания (реальные объекты мусора в COCO) ─────────────────
    39: 'plastic_bottle',   # bottle
    40: 'glass_bottle',     # wine glass
    41: 'can_cup',          # cup
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

    # ── Domain-gap: NavCamera смотрит вперёд, объекты видны нормально ───────
    # Оставляем только классы с высокой вероятностью ложного срабатывания
    # на наши конкретные объекты (банки, коробки).
    28: 'cardboard_paper',  # suitcase → картонная коробка (похожая форма)
    24: 'cardboard_paper',  # backpack → мешок мусора/пакет
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
        default_model = os.path.join(pkg, 'models', 'yolov8s_trash.pt')
        fallback_model = os.path.join(pkg, 'models', 'yolov8n_trash.pt')

        self.declare_parameter('model_path',  default_model)
        self.declare_parameter('conf_thresh', 0.35)
        self.declare_parameter('merge_dist',  1.2)
        self.declare_parameter('detect_rate', 4.0)
        self.declare_parameter('camera_mode', 'rgbd')
        self.declare_parameter('log_path', '/tmp/trash_detections.jsonl')
        self.declare_parameter('save_crops', True)
        self.declare_parameter('min_depth', 0.1)
        self.declare_parameter('max_depth', 10.0)
        self.declare_parameter('class_conf_overrides', '')
        self.declare_parameter('spawn_x',     0.0)
        self.declare_parameter('spawn_y',    -2.0)

        self._camera_mode = str(self.get_parameter('camera_mode').value)
        if self._camera_mode not in CAMERA_PROFILES:
            self.get_logger().warn(
                f'camera_mode={self._camera_mode!r} неизвестен, используем rgbd')
            self._camera_mode = 'rgbd'
        profile = CAMERA_PROFILES[self._camera_mode]

        self.declare_parameter('rgb_topic', profile['rgb_topic'])
        self.declare_parameter('depth_topic', profile['depth_topic'])
        self.declare_parameter('camera_info_topic', profile['camera_info_topic'])

        self._conf  = self.get_parameter('conf_thresh').value
        self._merge = self.get_parameter('merge_dist').value
        self._rate  = self.get_parameter('detect_rate').value
        self._spawn_x = self.get_parameter('spawn_x').value
        self._spawn_y = self.get_parameter('spawn_y').value
        self._log_path = str(self.get_parameter('log_path').value)
        self._save_crops = bool(self.get_parameter('save_crops').value)
        self._min_depth = float(self.get_parameter('min_depth').value)
        self._max_depth = float(self.get_parameter('max_depth').value)
        self._class_conf = self._parse_class_conf(
            str(self.get_parameter('class_conf_overrides').value)
        )
        self._rgb_topic = str(self.get_parameter('rgb_topic').value)
        self._depth_topic = str(self.get_parameter('depth_topic').value)
        self._camera_info_topic = str(self.get_parameter('camera_info_topic').value)
        self._cam_tx = float(profile['tx'])
        self._cam_tz = float(profile['tz'])
        self._cam_pitch = float(profile['pitch'])
        self._cam_fov = float(profile['fov'])
        self._localization = str(profile['localization'])
        self._camera_frame_id = str(profile['frame_id'])

        # ── Загрузка YOLOv8 ────────────────────────────────────────────── #
        model_path = str(self.get_parameter('model_path').value) or default_model
        if model_path == default_model and not os.path.isfile(model_path) and os.path.isfile(fallback_model):
            self.get_logger().warn(
                f'Основная yolov8s-модель не найдена: {default_model}; '
                f'временно использую старую baseline-модель: {fallback_model}'
            )
            model_path = fallback_model
        self._model = None
        self._is_coco = True  # до загрузки — безопасный default
        if _YOLO_OK:
            if os.path.isfile(model_path):
                self.get_logger().info(f'Загружаю YOLOv8: {model_path}')
                self._model = YOLO(model_path)
                self._model.fuse()
                # Определяем тип модели: COCO (80 кл, есть 'person') → фильтр по карте.
                # TACO/fine-tuned (нет 'person') → все классы = мусор.
                self._is_coco = 'person' in self._model.names.values()
                mode = 'COCO (фильтр по карте)' if self._is_coco else 'TACO/fine-tuned (все классы = мусор)'
                self.get_logger().info(f'YOLOv8 готов, режим: {mode}')
            else:
                self.get_logger().warn(
                    f'Файл модели не найден: {model_path}\n'
                    f'  Обучите YOLOv8s через scripts/train_yolo.py\n'
                    f'  или укажите свежий best.pt в trash_detector.model_path'
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
        self._last_registered_id: int | None = None
        self._frame_seq = 0

        self._log_fp = None
        if self._log_path:
            log_dir = os.path.dirname(self._log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            self._log_fp = open(self._log_path, 'a', buffering=1)
            self.get_logger().info(f'JSONL журнал детекций: {self._log_path}')

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
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        if self._camera_info_topic:
            self.create_subscription(CameraInfo, self._camera_info_topic, self._caminfo_cb, 1)
        self.create_subscription(Image, self._rgb_topic, self._rgb_cb, 10)
        if self._depth_topic:
            self.create_subscription(Image, self._depth_topic, self._depth_cb, 10)

        # ── Таймеры ────────────────────────────────────────────────────── #
        detect_period = 1.0 / max(0.5, self._rate)
        self.create_timer(detect_period, self._detect_timer)
        self.create_timer(5.0,           self._report_timer)

        self.get_logger().info(
            f'Detector: conf={self._conf}, merge={self._merge}м, '
            f'rate={self._rate}Гц, camera_mode={self._camera_mode}, '
            f'rgb={self._rgb_topic}, depth={self._depth_topic or "off"}'
        )

    @staticmethod
    def _parse_class_conf(raw: str) -> dict[str, float]:
        """Парсит строку вида 'cigarette_butt:0.2,plastic_bottle:0.45'."""
        overrides: dict[str, float] = {}
        for item in raw.split(','):
            item = item.strip()
            if not item:
                continue
            if ':' not in item:
                continue
            name, value = item.split(':', 1)
            name = name.strip().lower().replace(' ', '_').replace('-', '_')
            try:
                overrides[name] = float(value)
            except ValueError:
                continue
        return overrides

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
        if self._model is None or not self._odom_ok:
            return

        # Если camera_info так и не пришёл — считаем интринсики из параметров SDF.
        if self._K is None:
            fov, w, h = self._cam_fov, 640.0, 480.0
            fx = (w / 2.0) / math.tan(fov / 2.0)
            self._K = np.array(
                [[fx, 0.0, w / 2.0],
                 [0.0, fx, h / 2.0],
                 [0.0, 0.0, 1.0]], dtype=np.float64)
            self.get_logger().warn(
                f'camera_info не получен — используем {self._camera_mode} K: '
                f'fov={fov:.2f}, fx=fy={fx:.1f}, cx={w/2:.0f}, cy={h/2:.0f}'
            )

        with self._img_lock:
            if self._latest_rgb is None:
                return
            rgb   = self._latest_rgb.copy()
            # Depth необязателен: без него YOLO всё равно рисует bbox на кадре,
            # но пространственные маркеры на карте не выставляются.
            depth = self._latest_depth.copy() if self._latest_depth is not None else None
            rgb_stamp = self._rgb_stamp

        t0 = time.perf_counter()
        results = self._model(rgb, conf=self._conf, verbose=False)
        inference_ms = (time.perf_counter() - t0) * 1000.0
        if not results:
            return

        det = results[0]
        annotated = det.plot()   # кадр с нарисованными bbox
        self._frame_seq += 1
        frame_seq = self._frame_seq

        found_new = False
        for box in det.boxes:
            cls_id   = int(box.cls[0].item())
            conf_val = float(box.conf[0].item())
            label    = det.names[cls_id]

            if self._is_coco:
                # Базовая COCO: принимаем только классы из карты мусора
                if cls_id not in COCO_TRASH_MAP:
                    continue
                category = COCO_TRASH_MAP[cls_id]
            else:
                # Fine-tuned TACO: все классы = мусор, категория = имя класса
                category = label.lower().replace(' ', '_').replace('-', '_')

            class_threshold = self._class_conf.get(
                category,
                self._class_conf.get(
                    label.lower().replace(' ', '_').replace('-', '_'),
                    self._conf,
                )
            )
            if conf_val < class_threshold:
                continue

            # Центр bbox в пикселях
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            u = (x1 + x2) / 2.0
            v = (y1 + y2) / 2.0

            d = None
            wx, wy = None, None
            if self._localization == 'depth':
                d = self._sample_depth(depth, u, v) if depth is not None else None
                if d is not None:
                    wx, wy = self._pixel_to_world(u, v, d)
            elif self._localization == 'ground_plane':
                wx, wy = self._pixel_to_world_ground(u, v)

            world_xy = (wx, wy) if wx is not None else None
            if wx is not None:
                if self._register(wx, wy, category, label, conf_val,
                                  annotated, (u, v, x1, y1, x2, y2),
                                  depth=d,
                                  inference_ms=inference_ms,
                                  frame_seq=frame_seq):
                    found_new = True
            object_id = self._last_registered_id if wx is not None else None
            self._draw_detection_info(
                annotated,
                label=label,
                category=category,
                conf=conf_val,
                bbox_xyxy=(x1, y1, x2, y2),
                depth=d,
                world=world_xy,
                object_id=object_id,
                inference_ms=inference_ms,
                frame_seq=frame_seq,
            )
            self._log_detection(
                label=label,
                category=category,
                conf=conf_val,
                bbox=(x1, y1, x2, y2),
                depth=d,
                world=world_xy,
                object_id=object_id,
                inference_ms=inference_ms,
                frame_seq=frame_seq,
                frame_stamp=rgb_stamp,
            )

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
        valid = roi[
            np.isfinite(roi)
            & (roi >= self._min_depth)
            & (roi <= self._max_depth)
        ]
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

          3. SDF camera → body: t=(cam_tx, 0, cam_tz), Ry(cam_pitch)
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

        X_body, Y_body, _z_body = self._camera_point_to_body(X_sdf, Y_sdf, Z_sdf)

        # Шаг 4: world frame
        return self._body_to_world(X_body, Y_body)

    def _pixel_to_world_ground(self, u: float, v: float) -> tuple[float | None, float | None]:
        """Пиксель → пересечение луча камеры с плоскостью пола."""
        if self._K is None:
            return None, None

        fx = self._K[0, 0]; fy = self._K[1, 1]
        cx = self._K[0, 2]; cy = self._K[1, 2]

        x_opt = (u - cx) / fx
        y_opt = (v - cy) / fy
        # Луч в SDF camera frame при Z_opt=1.
        ray_x_sdf = 1.0
        ray_y_sdf = -x_opt
        ray_z_sdf = -y_opt

        dir_x, dir_y, dir_z = self._camera_vector_to_body(
            ray_x_sdf, ray_y_sdf, ray_z_sdf)
        origin_x, origin_y, origin_z = self._cam_tx, 0.0, self._cam_tz
        floor_z_body = -BODY_Z
        if dir_z >= -1e-6:
            return None, None

        t = (floor_z_body - origin_z) / dir_z
        if t <= 0.0:
            return None, None

        x_body = origin_x + t * dir_x
        y_body = origin_y + t * dir_y
        return self._body_to_world(x_body, y_body)

    def _camera_vector_to_body(self, x_sdf: float, y_sdf: float,
                               z_sdf: float) -> tuple[float, float, float]:
        p = self._cam_pitch
        cp, sp = math.cos(p), math.sin(p)
        return (
            cp * x_sdf + sp * z_sdf,
            y_sdf,
            -sp * x_sdf + cp * z_sdf,
        )

    def _camera_point_to_body(self, x_sdf: float, y_sdf: float,
                              z_sdf: float) -> tuple[float, float, float]:
        x, y, z = self._camera_vector_to_body(x_sdf, y_sdf, z_sdf)
        return x + self._cam_tx, y, z + self._cam_tz

    def _body_to_world(self, x_body: float, y_body: float) -> tuple[float, float]:
        yaw = self._robot_yaw
        cy_r, sy_r = math.cos(yaw), math.sin(yaw)
        wx = self._robot_x + cy_r * x_body - sy_r * y_body
        wy = self._robot_y + sy_r * x_body + cy_r * y_body
        return wx, wy

    # ── Регистрация объектов ──────────────────────────────────────────────── #

    def _register(self, wx: float, wy: float, category: str,
                  label: str, conf: float,
                  frame_bgr: np.ndarray | None = None,
                  bbox: tuple | None = None,
                  depth: float | None = None,
                  inference_ms: float | None = None,
                  frame_seq: int | None = None) -> bool:
        """
        Добавляет объект в базу или обновляет уверенность существующего.
        Возвращает True, если добавлен НОВЫЙ объект.
        При добавлении нового объекта сохраняет кроп bbox на диск.
        """
        # Слияние: ищем ближайший уже известный объект
        self._last_registered_id = None
        for tid, obj in self._trash.items():
            if math.hypot(wx - obj['x'], wy - obj['y']) < self._merge:
                obj['x'] = 0.8 * obj['x'] + 0.2 * wx
                obj['y'] = 0.8 * obj['y'] + 0.2 * wy
                obj['count'] += 1
                if conf > obj['conf']:
                    obj['conf'] = conf
                self._last_registered_id = tid
                return False

        # Новый объект
        tid = self._trash_counter
        self._trash_counter += 1
        self._last_registered_id = tid
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
        if self._save_crops and frame_bgr is not None and bbox is not None:
            self._save_detection(
                tid=tid,
                label=label,
                category=category,
                conf=conf,
                frame_bgr=frame_bgr,
                bbox=bbox,
                depth=depth,
                world=(wx, wy),
                inference_ms=inference_ms,
                frame_seq=frame_seq,
            )

        return True

    @staticmethod
    def _safe_filename(value: str) -> str:
        safe = ''.join(
            ch if ch.isalnum() or ch in ('_', '-') else '_'
            for ch in str(value).strip().replace(' ', '_')
        )
        return safe or 'trash'

    def _detection_metadata(self, object_id: int | None,
                            label: str,
                            category: str,
                            conf: float,
                            bbox_xyxy: tuple[float, float, float, float],
                            depth: float | None,
                            world: tuple[float, float] | None,
                            inference_ms: float | None,
                            frame_seq: int | None) -> dict:
        x1, y1, x2, y2 = bbox_xyxy
        return {
            'saved_at': datetime.now().isoformat(timespec='seconds'),
            'object_id': object_id,
            'class': category,
            'label': label,
            'confidence': float(conf),
            'bbox_xyxy': [float(x1), float(y1), float(x2), float(y2)],
            'bbox_center': [float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)],
            'bbox_size_px': [float(x2 - x1), float(y2 - y1)],
            'depth_m': None if depth is None else float(depth),
            'world': None if world is None else {
                'x': float(world[0]),
                'y': float(world[1]),
            },
            'robot': {
                'x': float(getattr(self, '_robot_x', 0.0)),
                'y': float(getattr(self, '_robot_y', 0.0)),
                'yaw': float(getattr(self, '_robot_yaw', 0.0)),
            },
            'camera_mode': getattr(self, '_camera_mode', 'unknown'),
            'inference_ms': None if inference_ms is None else float(inference_ms),
            'frame_seq': frame_seq,
        }

    def _detection_stat_lines(self, object_id: int | None,
                              label: str,
                              category: str,
                              conf: float,
                              bbox_xyxy: tuple[float, float, float, float],
                              depth: float | None,
                              world: tuple[float, float] | None,
                              inference_ms: float | None,
                              frame_seq: int | None,
                              compact: bool = False) -> list[str]:
        x1, y1, x2, y2 = bbox_xyxy
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        obj = 'n/a' if object_id is None else str(object_id)
        depth_text = 'n/a' if depth is None else f'{depth:.2f} m'
        world_text = (
            'n/a'
            if world is None
            else f'x={world[0]:.2f} y={world[1]:.2f} m'
        )
        inf_text = 'n/a' if inference_ms is None else f'{inference_ms:.1f} ms'
        if compact:
            return [
                f'#{obj} {category} ({conf:.2f})',
                f'world: {world_text}',
                f'depth: {depth_text}',
                f'bbox: {x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}',
                f'inf: {inf_text} frame: {frame_seq}',
            ]
        return [
            f'object_id: {obj}',
            f'class: {category}',
            f'label: {label}',
            f'confidence: {conf:.3f}',
            f'bbox_xyxy: {x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}',
            f'bbox_center_px: {cx:.0f}, {cy:.0f}',
            f'bbox_size_px: {x2 - x1:.0f} x {y2 - y1:.0f}',
            f'depth: {depth_text}',
            f'world: {world_text}',
            f'robot: x={getattr(self, "_robot_x", 0.0):.2f} '
            f'y={getattr(self, "_robot_y", 0.0):.2f} '
            f'yaw={getattr(self, "_robot_yaw", 0.0):.2f}',
            f'camera_mode: {getattr(self, "_camera_mode", "unknown")}',
            f'inference: {inf_text}',
            f'frame_seq: {frame_seq}',
        ]

    def _draw_detection_info(self, frame: np.ndarray,
                             label: str,
                             category: str,
                             conf: float,
                             bbox_xyxy: tuple[float, float, float, float],
                             depth: float | None,
                             world: tuple[float, float] | None,
                             object_id: int | None,
                             inference_ms: float | None,
                             frame_seq: int | None) -> None:
        """Рисует bbox и компактную статистику прямо на кадре."""
        if frame is None or frame.size == 0:
            return
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox_xyxy
        bx1 = int(np.clip(x1, 0, max(0, w - 1)))
        by1 = int(np.clip(y1, 0, max(0, h - 1)))
        bx2 = int(np.clip(x2, 0, max(0, w - 1)))
        by2 = int(np.clip(y2, 0, max(0, h - 1)))
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)

        lines = self._detection_stat_lines(
            object_id=object_id,
            label=label,
            category=category,
            conf=conf,
            bbox_xyxy=bbox_xyxy,
            depth=depth,
            world=world,
            inference_ms=inference_ms,
            frame_seq=frame_seq,
            compact=True,
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.42
        thickness = 1
        line_height = 16
        pad = 5
        max_lines = max(1, (h - 2 * pad) // line_height)
        lines = lines[:max_lines]
        text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
        panel_w = min(w, max(size[0] for size in text_sizes) + 2 * pad)
        panel_h = min(h, line_height * len(lines) + 2 * pad)
        px = max(0, min(bx1, w - panel_w))
        py_above = by1 - panel_h - 4
        py = py_above if py_above >= 0 else min(max(0, by2 + 4), max(0, h - panel_h))

        overlay = frame.copy()
        cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (16, 16, 16), -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0.0, frame)
        for idx, line in enumerate(lines):
            y = py + pad + (idx + 1) * line_height - 3
            cv2.putText(frame, line, (px + pad, y), font, font_scale,
                        (255, 255, 255), thickness, cv2.LINE_AA)

    def _make_detection_card(self, crop: np.ndarray, lines: list[str]) -> np.ndarray:
        """Создает crop-карточку: слева объект, справа подробная статистика."""
        crop_vis = crop.copy()
        ch, cw = crop_vis.shape[:2]
        if ch == 0 or cw == 0:
            return crop_vis

        max_side = max(ch, cw)
        if max_side < 180:
            scale = 180.0 / max_side
            crop_vis = cv2.resize(
                crop_vis,
                (max(1, int(cw * scale)), max(1, int(ch * scale))),
                interpolation=cv2.INTER_LINEAR,
            )
        elif max_side > 520:
            scale = 520.0 / max_side
            crop_vis = cv2.resize(
                crop_vis,
                (max(1, int(cw * scale)), max(1, int(ch * scale))),
                interpolation=cv2.INTER_AREA,
            )

        ch, cw = crop_vis.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.50
        thickness = 1
        line_height = 21
        pad = 14
        panel_w = 470
        panel_h = line_height * len(lines) + 2 * pad
        card_h = max(ch, panel_h)
        card = np.full((card_h, cw + panel_w, 3), 245, dtype=np.uint8)
        card[:ch, :cw] = crop_vis
        cv2.rectangle(card, (cw, 0), (cw + panel_w, card_h), (24, 24, 24), -1)
        for idx, line in enumerate(lines):
            y = pad + (idx + 1) * line_height - 4
            cv2.putText(card, line[:62], (cw + pad, y), font, font_scale,
                        (245, 245, 245), thickness, cv2.LINE_AA)
        return card

    def _save_detection(self, tid: int, label: str, category: str, conf: float,
                        frame_bgr: np.ndarray, bbox: tuple,
                        depth: float | None = None,
                        world: tuple[float, float] | None = None,
                        inference_ms: float | None = None,
                        frame_seq: int | None = None):
        """Сохраняет кадр, crop-card и JSON-метаданные в /tmp/trash_detected/."""
        ts = datetime.now().strftime('%H%M%S_%f')[:10]
        safe_label = self._safe_filename(label)
        _u, _v, x1, y1, x2, y2 = bbox
        bbox_xyxy = (x1, y1, x2, y2)
        lines = self._detection_stat_lines(
            object_id=tid,
            label=label,
            category=category,
            conf=conf,
            bbox_xyxy=bbox_xyxy,
            depth=depth,
            world=world,
            inference_ms=inference_ms,
            frame_seq=frame_seq,
        )

        # Полный кадр с bbox и статистикой.
        full_frame = frame_bgr.copy()
        self._draw_detection_info(
            full_frame,
            label=label,
            category=category,
            conf=conf,
            bbox_xyxy=bbox_xyxy,
            depth=depth,
            world=world,
            object_id=tid,
            inference_ms=inference_ms,
            frame_seq=frame_seq,
        )
        full_path = os.path.join(
            self._save_dir, f'{ts}_id{tid:03d}_{safe_label}_full.jpg')
        cv2.imwrite(full_path, full_frame)

        # Кроп bbox (с небольшим отступом) и отдельная карточка с текстовой панелью.
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
            card = self._make_detection_card(crop, lines)
            card_path = os.path.join(
                self._save_dir, f'{ts}_id{tid:03d}_{safe_label}_card.jpg')
            cv2.imwrite(card_path, card)

        meta_path = os.path.join(
            self._save_dir, f'{ts}_id{tid:03d}_{safe_label}_meta.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(
                self._detection_metadata(
                    object_id=tid,
                    label=label,
                    category=category,
                    conf=conf,
                    bbox_xyxy=bbox_xyxy,
                    depth=depth,
                    world=world,
                    inference_ms=inference_ms,
                    frame_seq=frame_seq,
                ),
                f,
                ensure_ascii=False,
                indent=2,
            )
        self.get_logger().info(f'  → сохранено: {full_path}')

    @staticmethod
    def _stamp_to_float(stamp) -> float | None:
        if stamp is None:
            return None
        sec = getattr(stamp, 'sec', None)
        nanosec = getattr(stamp, 'nanosec', None)
        if sec is None or nanosec is None:
            return None
        return float(sec) + float(nanosec) * 1e-9

    def _log_detection(self, label: str, category: str, conf: float,
                       bbox: tuple[float, float, float, float],
                       depth: float | None,
                       world: tuple[float, float] | None,
                       object_id: int | None,
                       inference_ms: float,
                       frame_seq: int,
                       frame_stamp) -> None:
        """Пишет одну строку JSONL на bbox для воспроизводимой оценки демо."""
        if self._log_fp is None:
            return
        x1, y1, x2, y2 = bbox
        record = {
            'timestamp_wall': time.time(),
            'timestamp_ros': self._stamp_to_float(frame_stamp),
            'frame_seq': int(frame_seq),
            'object_id': object_id,
            'class': category,
            'label': label,
            'confidence': float(conf),
            'bbox_xyxy': [float(x1), float(y1), float(x2), float(y2)],
            'bbox_center': [float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)],
            'depth_m': None if depth is None else float(depth),
            'world': None if world is None else {
                'x': float(world[0]),
                'y': float(world[1]),
            },
            'robot': {
                'x': float(self._robot_x),
                'y': float(self._robot_y),
                'yaw': float(self._robot_yaw),
            },
            'camera_mode': self._camera_mode,
            'inference_ms': float(inference_ms),
        }
        self._log_fp.write(json.dumps(record, ensure_ascii=False) + '\n')

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
        msg.header.frame_id = self._camera_frame_id
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
        if getattr(node, '_log_fp', None) is not None:
            node._log_fp.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
