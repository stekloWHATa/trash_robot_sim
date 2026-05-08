# trash_robot_sim

**Система локализации и идентификации бытового мусора в рабочей сцене мобильного робота**

ROS 2/Gazebo Harmonic симуляция мобильного робота для демонстрации главного
дипломного сценария: робот едет по подготовленной сцене, YOLOv8 распознает
мусор на RGB-кадре, depth-камера дает расстояние, а найденные объекты
появляются на карте/RViz как 3D-маркеры.

Навигация с A*/boustrophedon оставлена в проекте как будущая доп. фича. Для
актуального MVP основной запуск - `detection_demo.launch.py`: без автономного
объезда препятствий, с коротким scripted motion для записи видео.

## Архитектура

```
Gazebo Harmonic
    |
    | ros_gz_bridge
    v
/cmd_vel <------------- scripted_motion / teleop
/odom -----------------> trash_detector
/scan -----------------> map_builder ----------> /map
/rgbd/image/image -----> trash_detector -------> /detections_img
/rgbd/image/depth_image -> trash_detector -----> /trash_markers
/rgbd/image/camera_info -> trash_detector -----> /trash_report + JSONL log
/tf, /joint_states ----> robot_state_publisher -> RViz
```

## Узлы

| Узел | Файл | Назначение |
|---|---|---|
| `scripted_motion` | `scripts/scripted_motion.py` | Простой повторяемый маршрут для видеодемо |
| `astar_navigator` | `scripts/navigator.py` | Future work: A*, покрытие области, pure pursuit |
| `map_builder` | `scripts/map_builder.py` | OccupancyGrid по лидару через log-odds |
| `trash_detector` | `scripts/detector.py` | YOLOv8-детекция, RGBD/ground-plane локализация, маркеры и отчёт |
| `robot_state_publisher` | системный | `/robot_description` и TF |
| `gz_bridge` | системный | Мост Gazebo <-> ROS 2 |

## Основные топики

| Топик | Тип | Описание |
|---|---|---|
| `/cmd_vel` | `geometry_msgs/Twist` | Команды скорости робота |
| `/odom` | `nav_msgs/Odometry` | Одометрия из Gazebo |
| `/scan` | `sensor_msgs/LaserScan` | Лидар |
| `/map` | `nav_msgs/OccupancyGrid` | Карта занятости |
| `/rgbd/image/image` | `sensor_msgs/Image` | RGB-кадр для YOLOv8 |
| `/rgbd/image/depth_image` | `sensor_msgs/Image` | Depth-карта для 3D-локализации |
| `/detections_img` | `sensor_msgs/Image` | Кадр с bbox |
| `/trash_markers` | `visualization_msgs/MarkerArray` | Найденные объекты в RViz |
| `/trash_report` | `std_msgs/String` | Текстовый отчёт |
| `/scan_area` | `geometry_msgs/Polygon` | Область покрытия |
| `/goal_pose` | `geometry_msgs/PoseStamped` | Разовая цель из RViz |

## Быстрый запуск

Главный сценарий для видео:

```bash
cd ~/ros2_ws
colcon build --packages-select trash_robot_sim
source install/setup.bash
ros2 launch trash_robot_sim detection_demo.launch.py scripted_motion:=true
```

В другом терминале:

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch trash_robot_sim rviz.launch.py
```

Что смотреть/записывать:

- Gazebo: робот проезжает мимо подготовленного мусора в
  `worlds/detection_demo_world.sdf`;
- RViz: `/map`, `/trash_markers`, `/detections_img`;
- лог детектора: `/tmp/trash_detections.jsonl`;
- фото новых объектов: `/tmp/trash_detected` (`*_full.jpg`, `*_crop.jpg`,
  `*_card.jpg`, `*_meta.json`).

Модели мусора подключены как скачанные `.glb` mesh-assets из Poly Pizza
(`models/trash/ASSET_SOURCES.md`): Kenney/Quaternius CC0 и один CC-BY окурок
Poly by Google.

Статичный запуск без scripted motion:

```bash
ros2 launch trash_robot_sim detection_demo.launch.py scripted_motion:=false
```

После записи прогона можно посчитать честный отчет по ground truth сцены:

```bash
python3 scripts/evaluate_detection_run.py \
  --ground-truth config/trash_ground_truth.yaml \
  --detections /tmp/trash_detections.jsonl
```

Автоматически сохранить кадры из Gazebo/детектора без ручных скриншотов:

```bash
ros2 run trash_robot_sim run_demo_capture.py \
  --output-dir /tmp/trash_demo_capture \
  --max-frames 30
```

Старый полный запуск с навигатором остается доступен:

```bash
ros2 launch trash_robot_sim gazebo.launch.py
```

## YOLOv8-датасет

Для финального обучения под 80-85% используем MVP-таксономию
`config/trash_classes_mvp.yaml`:

- `cigarette_butt`
- `plastic_bottle`
- `aluminum_can`
- `plastic_bag`
- `cardboard_box`
- `paper_packaging`

Полная таксономия на 8 классов остается в `config/trash_classes.yaml`, но
`glass_bottle` и `other_trash` лучше подключать после стабильного результата на
MVP-классах.

Подготовить новый merged-набор без запуска долгого обучения:

```bash
python3 scripts/train_yolo.py --skip-download --prepare-only
```

По умолчанию результат пишется в `data/merged_v2`, чтобы не трогать старый
`data/merged`. Скрипт применяет mapping классов, пропускает ненужные классы и
сохраняет `merge_report.yaml`. Если входной YOLO-файл содержит segmentation
polygon (`class x1 y1 x2 y2 ...`), он сворачивается в bbox для YOLO detect.

Аудит YOLO-набора:

```bash
python3 scripts/audit_dataset.py data/merged_v2/data.yaml
```

Конвертация COCO/TACO-аннотаций в YOLO:

```bash
python3 scripts/train_yolo.py \
  --skip-download \
  --coco-json /path/to/annotations.json \
  --coco-images /path/to/images \
  --coco-output data/taco_yolo \
  --coco-split auto \
  --prepare-only
```

Затем добавить конвертированный набор в общий merge:

```bash
python3 scripts/train_yolo.py \
  --skip-download \
  --extra-dataset data/taco_yolo \
  --prepare-only
```

Долгое обучение запускать только после аудита:

```bash
python3 scripts/check_dataset_readiness.py data/taco_only_detection_v1/data.yaml

python3 scripts/train_yolo.py \
  --skip-download \
  --extra-dataset data/taco_yolo \
  --output-dir data/taco_only_detection_v1 \
  --epochs 120 \
  --imgsz 960 \
  --base-model models/yolov8s.pt \
  --output-model models/yolov8s_trash.pt \
  --run-name taco_only_yolov8s_img960
```

## Параметры

Все параметры находятся в `config/params.yaml`.

Ключевые параметры `trash_detector`:

| Параметр | Значение | Описание |
|---|---:|---|
| `model_path` | `""` | Пусто = `share/models/yolov8s_trash.pt`; если ее нет, временный fallback на `yolov8n_trash.pt` |
| `conf_thresh` | `0.35` | Порог уверенности YOLO |
| `merge_dist` | `1.2` | Радиус слияния повторных 3D-детекций |
| `detect_rate` | `4.0` | Частота инференса |
| `camera_mode` | `rgbd` | RGB и depth из одной RGBD-камеры |
| `log_path` | `/tmp/trash_detections.jsonl` | JSONL-журнал bbox/depth/world/latency |
| `save_crops` | `true` | Сохранять кадр, crop, карточку объекта и JSON-метаданные |
| `min_depth`/`max_depth` | `0.10`/`10.0` | Фильтр валидной depth-карты |
| `class_conf_overrides` | `""` | Порог по классам, например `cigarette_butt:0.2` |

`camera_mode=nav_ground_plane` использует `/camera/image` и локализует объект
пересечением луча камеры с плоскостью пола. Основной режим для дипломной
демонстрации - `rgbd`.

## Тесты

Быстрые unit-тесты без запуска Gazebo:

```bash
python3 -m pytest tests -q
```

Покрыто:

- A* и boustrophedon-навигация;
- безопасность маршрутов около препятствий;
- merge pipeline для YOLO-датасетов;
- COCO/TACO -> YOLO conversion;
- аудит YOLO-разметки;
- геометрия depth/ground-plane локализации детектора;
- smoke-тест detection demo assets/launch;
- matching ground truth ↔ detector log.

## Структура

```
trash_robot_sim/
├── config/
│   ├── params.yaml
│   ├── trash_ground_truth.yaml
│   └── trash_classes.yaml
├── docs/
│   ├── datasets.md
│   ├── experiments.md
│   └── diploma_status_and_plan.md
├── launch/
├── models/
├── rviz/
├── scripts/
│   ├── audit_dataset.py
│   ├── detector.py
│   ├── evaluate_detection_run.py
│   ├── map_builder.py
│   ├── navigator.py
│   ├── scripted_motion.py
│   └── train_yolo.py
├── tests/
├── urdf/
└── worlds/
```
