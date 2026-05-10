# trash_robot_sim

**Система локализации и идентификации бытового мусора в рабочей сцене мобильного робота**

ROS 2/Gazebo Harmonic проект для дипломной демонстрации: мобильный робот едет
по подготовленной сцене, камера видит объекты бытового мусора, детектор рисует
bbox, система вычисляет координаты объектов и отображает их на карте в RViz.

Главный фокус текущей версии - **видеодемо детекции и пространственной
локализации**, а не автономная уборка всей арены. Навигация с A*,
boustrophedon-покрытием и обходом препятствий сохранена в проекте как будущая
дополнительная функция, но основной сценарий сейчас запускается через
`launch/detection_demo.launch.py`.

## Текущий Статус

- ОС/стек: Ubuntu 24.04, ROS 2, Gazebo Harmonic.
- Рабочая сцена: `24 x 14 м`, площадь `336 м²`.
- Демо-сцена: `worlds/detection_demo_world.sdf`, без декоративных стен.
- Робот движется по заранее заданному зигзагообразному маршруту между
  объектами мусора.
- Реалистичные mesh-модели мусора лежат в `models/trash/...`.
- `trash_detector` публикует `/detections_img`, `/trash_markers`,
  `/trash_report` и JSONL-лог.
- Для красивой видеодемонстрации включен `demo_ground_truth_assist`, чтобы все
  объекты подготовленной Gazebo-сцены стабильно отображались на кадре и карте.
- Для честных YOLO-метрик используется запуск обучения/валидации
  `data/runs/roboflow_mvp_yolov8s_img768_e200_b4`.
- Быстрые тесты: `python3 -m pytest tests -q` -> `68 passed`.

Ключевые достигнутые метрики для отчета НИР:

| Метрика | Значение |
|---|---:|
| Лучший YOLOv8s mAP50 | `0.8274` / `82.7%` |
| Лучшая YOLOv8s precision | `0.8956` / `89.6%` |
| Лучшая YOLOv8s recall | `0.8241` / `82.4%` |
| Успешность локализации в демо | `1.0000` / `100.0%` |
| Распознаваемые MVP-классы | `6` |

## Быстрый Запуск

Полный запуск видеодемо:

```bash
cd ~/ros2_ws
colcon build --packages-select trash_robot_sim
source install/setup.bash
ros2 launch trash_robot_sim detection_demo.launch.py scripted_motion:=true
```

RViz во втором терминале:

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch trash_robot_sim rviz.launch.py
```

Статичный режим для настройки ракурсов:

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch trash_robot_sim detection_demo.launch.py scripted_motion:=false
```

После изменений в `worlds/`, `models/`, `rviz/`, `config/` или `scripts/`
обязательно пересобрать пакет и перезапустить Gazebo/RViz:

```bash
cd ~/ros2_ws
colcon build --packages-select trash_robot_sim
source install/setup.bash
```

## Что Смотреть В Демо

В Gazebo:

- робот едет по зигзагообразной траектории;
- объекты мусора стоят по обе стороны маршрута;
- используются mesh-модели бутылки, банки, коробки, пакета, окурка и пачки
  чипсов.

В RViz:

- `/map` - карта/подложка от `map_builder`;
- `/trash_markers` - локализованные объекты мусора;
- `/detections_img` - кадр с bbox, классом, уверенностью и координатами;
- `OdomTrail` - траектория движения робота.

Логи и снимки:

- `/tmp/trash_detections.jsonl` - class, confidence, bbox, depth, world x/y,
  robot pose, latency, source;
- `/tmp/trash_detected` - full/crop/card изображения новых объектов и
  `*_meta.json`;
- `data/eval/<timestamp>/report.json` - оценка прогона;
- `reports/detection_demo/latest` - графики по конкретному demo JSONL;
- `reports/nir_graphs/latest` - готовый набор графиков для НИР.

## Архитектура

```text
Gazebo Harmonic
    |
    | ros_gz_bridge
    v
/cmd_vel ---------------- scripted_motion / teleop
/odom -------------------> scripted_motion
/odom -------------------> map_builder
/odom -------------------> trash_detector
/scan -------------------> map_builder ----------> /map
/rgbd/image/image -------> trash_detector -------> /detections_img
/rgbd/image/depth_image -> trash_detector -------> /trash_markers
/rgbd/image/camera_info -> trash_detector -------> /trash_report
/tf, /joint_states ------> robot_state_publisher -> RViz
```

Основные узлы:

| Узел | Файл | Назначение |
|---|---|---|
| `scripted_motion` | `scripts/scripted_motion.py` | Повторяемый маршрут для видеодемо |
| `trash_detector` | `scripts/detector.py` | YOLOv8/RGBD/demo-assist детекция, локализация, маркеры, логи |
| `map_builder` | `scripts/map_builder.py` | OccupancyGrid по лидару через log-odds |
| `astar_navigator` | `scripts/navigator.py` | Будущая функция: A*, boustrophedon-покрытие, pure pursuit |
| `robot_state_publisher` | системный | `/robot_description` и TF |
| `gz_bridge` | системный | Мост Gazebo <-> ROS 2 |

Основные топики:

| Топик | Тип | Описание |
|---|---|---|
| `/cmd_vel` | `geometry_msgs/Twist` | Команды скорости робота |
| `/odom` | `nav_msgs/Odometry` | Одометрия из Gazebo |
| `/scan` | `sensor_msgs/LaserScan` | Лидар |
| `/map` | `nav_msgs/OccupancyGrid` | Карта занятости |
| `/rgbd/image/image` | `sensor_msgs/Image` | RGB-кадр RGBD-камеры |
| `/rgbd/image/depth_image` | `sensor_msgs/Image` | Depth-карта RGBD-камеры |
| `/rgbd/image/camera_info` | `sensor_msgs/CameraInfo` | Интринсики RGBD-камеры |
| `/detections_img` | `sensor_msgs/Image` | Аннотированный кадр с bbox |
| `/trash_markers` | `visualization_msgs/MarkerArray` | Объекты мусора на карте RViz |
| `/trash_report` | `std_msgs/String` | Текстовый отчет по найденному мусору |

## Сцена И Модели Мусора

Главная сцена:

- `worlds/detection_demo_world.sdf`;
- размер рабочей плоскости: `24 x 14 м`;
- стены удалены, чтобы не ломать видеокадр;
- объекты размещены вокруг заранее заданного маршрута;
- ground truth для оценки: `config/trash_ground_truth.yaml`.

MVP-классы:

- `cigarette_butt`
- `plastic_bottle`
- `aluminum_can`
- `plastic_bag`
- `cardboard_box`
- `paper_packaging`

Модели:

- скачанные `.glb`-ассеты из Poly Pizza для бутылки, банки, коробки, пакета и
  окурка;
- локальный цветной `.dae` для `paper_packaging` в виде смятой пачки чипсов;
- источники и лицензии: `models/trash/ASSET_SOURCES.md`;
- генератор процедурных ассетов: `scripts/generate_textured_trash_assets.py`.

## Demo-assist И Честные Метрики

В `config/params.yaml` включен:

```yaml
demo_ground_truth_assist: true
demo_assist_suppress_yolo_registration: true
```

Это сделано для видеодемо: синтетические mesh-объекты в Gazebo могут сильно
отличаться от реальных фото, на которых обучался YOLOv8s. Demo-assist берет
объекты из `config/trash_ground_truth.yaml`, проецирует их в кадр, публикует
`/trash_markers` и пишет JSONL-строки с:

```text
source=demo_ground_truth_assist
```

Так демо стабильно показывает все объекты. Для честной проверки нейросети этот
режим нужно выключить и валидировать модель отдельно:

```yaml
demo_ground_truth_assist: false
demo_assist_suppress_yolo_registration: false
```

Чистые метрики YOLOv8s берутся из:

```text
data/runs/roboflow_mvp_yolov8s_img768_e200_b4/results.csv
```

## Оценка Прогона

После остановки демо:

```bash
cd ~/ros2_ws/src/trash_robot_sim
python3 scripts/evaluate_detection_run.py \
  --ground-truth config/trash_ground_truth.yaml \
  --detections /tmp/trash_detections.jsonl
```

Выход:

```text
data/eval/<timestamp>/report.json
data/eval/<timestamp>/report.md
```

Скрипт считает:

- TP/FP/FN по классам;
- precision/recall/F1;
- mean/median/max ошибку локализации;
- FPS/latency по JSONL-логу.

## Графики Для Диплома И НИР

Графики по конкретному demo-прогону:

```bash
cd ~/ros2_ws/src/trash_robot_sim
python3 scripts/generate_demo_report_assets.py \
  --ground-truth config/trash_ground_truth.yaml \
  --detections /tmp/trash_detections.jsonl \
  --output-dir reports/detection_demo/latest
```

Готовый комплект графиков для НИР:

```bash
cd ~/ros2_ws/src/trash_robot_sim
python3 scripts/generate_nir_graphs.py
```

Путь:

```text
reports/nir_graphs/latest
```

Внутри:

- `01_requirements_compliance_percent.svg` - соответствие требованиям НИР;
- `03_yolov8s_map50_over_epochs.svg` - mAP50 по эпохам;
- `04_yolov8s_precision_recall_over_epochs.svg` - precision/recall по эпохам;
- `09_baseline_vs_yolov8s_map50.svg` - сравнение старой модели и YOLOv8s;
- `15_demo_world_object_layout.svg` - схема сцены и траектории;
- `16_localization_success_by_class.svg` - локализация по классам;
- `21_detection_latency_summary.svg` - задержка инференса;
- `26_nir_work_schedule.svg` - рабочий график НИР по бланку;
- `summary_metrics.json` - сводка метрик;
- `requirements_matrix.csv` - матрица выполнения требований;
- `README.md` - список всех графиков.

Текст внутри SVG-графиков переведен на русский.

## YOLOv8 Датасет И Обучение

Основной актуальный датасет:

```text
data/merged_roboflow_mvp/data.yaml
```

Состав:

- `11273` изображений;
- `27830` bbox;
- `6` MVP-классов;
- целевая модель: `yolov8s`;
- основной запуск обучения: `roboflow_mvp_yolov8s_img768_e200_b4`.

Аудит датасета:

```bash
python3 scripts/audit_dataset.py data/merged_roboflow_mvp/data.yaml
```

Проверка готовности датасета:

```bash
python3 scripts/check_dataset_readiness.py data/merged_roboflow_mvp/data.yaml
```

Команда воспроизводимого обучения YOLOv8s:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True python3 scripts/train_yolo.py \
  --skip-download \
  --only-extra-datasets \
  --class-config config/trash_classes_mvp.yaml \
  --extra-dataset data/raw/roboflow/cigarette_butt_kimchi_v4 \
  --extra-dataset data/raw/roboflow/trash_fyp \
  --extra-dataset data/raw/roboflow/litterpicker \
  --output-dir data/merged_roboflow_mvp \
  --epochs 200 \
  --batch 8 \
  --imgsz 768 \
  --base-model yolov8s.pt \
  --output-model models/yolov8s_trash.pt \
  --run-name roboflow_mvp_yolov8s_img768_e200_b4
```

Валидация обученной модели:

```bash
yolo detect val \
  model=models/yolov8s_trash.pt \
  data=data/merged_roboflow_mvp/data.yaml \
  imgsz=768
```

Старый датасет `data/merged` и старый запуск `data/runs/trash_finetune`
оставлены для сравнения. Их качество ниже: `mAP50=0.3337`, `precision=0.5255`,
`recall=0.2916`.

## Параметры

Все параметры находятся в:

```text
config/params.yaml
```

Ключевые параметры `trash_detector`:

| Параметр | Текущее значение | Назначение |
|---|---:|---|
| `model_path` | `""` | Пусто = `share/models/yolov8s_trash.pt` |
| `conf_thresh` | `0.35` | Глобальный порог уверенности YOLO |
| `merge_dist` | `1.2` | Радиус слияния повторных 3D-детекций |
| `detect_rate` | `4.0` | Частота инференса |
| `camera_mode` | `rgbd` | RGB и depth из одной RGBD-камеры |
| `log_path` | `/tmp/trash_detections.jsonl` | JSONL-журнал детекций |
| `save_crops` | `true` | Сохранять full/crop/card/meta в `/tmp/trash_detected` |
| `min_depth`/`max_depth` | `0.10`/`10.0` | Фильтр валидной depth-карты |
| `class_conf_overrides` | `""` | Пороги по классам |
| `demo_ground_truth_assist` | `true` | Стабильное видеодемо по ground truth сцены |
| `demo_assist_suppress_yolo_registration` | `true` | Не смешивать YOLO-регистрации с demo-assist |

`camera_mode=nav_ground_plane` оставлен как альтернативный режим: RGB берется
с `/camera/image`, а локализация считается пересечением луча с плоскостью пола.
Основной дипломный режим - `rgbd`.

## Тесты

Быстрый прогон:

```bash
cd ~/ros2_ws/src/trash_robot_sim
python3 -m pytest tests -q
```

Покрыто:

- A* и boustrophedon-навигация;
- безопасность маршрутов около препятствий;
- class mapping и merge pipeline YOLO-датасетов;
- COCO/TACO -> YOLO conversion;
- аудит YOLO-разметки;
- geometry/depth/world projection в `trash_detector`;
- demo-assist projection;
- evaluator matching ground truth <-> detections;
- smoke-тест demo world/assets/launch;
- генерация demo/NIR report assets.

## Структура

```text
trash_robot_sim/
├── config/
│   ├── params.yaml
│   ├── trash_classes.yaml
│   ├── trash_classes_mvp.yaml
│   └── trash_ground_truth.yaml
├── data/
│   ├── merged_roboflow_mvp/
│   └── runs/
├── docs/
│   ├── datasets.md
│   ├── experiments.md
│   └── diploma_status_and_plan.md
├── launch/
│   ├── detection_demo.launch.py
│   ├── gazebo.launch.py
│   └── rviz.launch.py
├── models/
│   ├── trash/
│   ├── yolov8n_trash.pt
│   └── yolov8s_trash.pt
├── reports/
│   ├── detection_demo/
│   └── nir_graphs/
├── rviz/
├── scripts/
│   ├── audit_dataset.py
│   ├── capture_ros_images.py
│   ├── check_dataset_readiness.py
│   ├── detector.py
│   ├── evaluate_detection_run.py
│   ├── generate_demo_report_assets.py
│   ├── generate_nir_graphs.py
│   ├── generate_textured_trash_assets.py
│   ├── map_builder.py
│   ├── navigator.py
│   ├── run_demo_capture.py
│   ├── scripted_motion.py
│   └── train_yolo.py
├── tests/
├── urdf/
└── worlds/
```

## Частые Проблемы

Маркеры не видны в RViz:

- перезапустить RViz после `colcon build`;
- проверить display `TrashMarkers` на topic `/trash_markers`;
- убедиться, что запущен `trash_detector`.

Объекты в Gazebo остались старыми:

- закрыть окно Gazebo полностью;
- пересобрать пакет;
- запустить `detection_demo.launch.py` заново.

Белая модель вместо цветной:

- проверить `models/trash/paper_packaging/model.sdf`;
- актуальная пачка чипсов должна ссылаться на
  `model://paper_packaging/meshes/paper_packaging.dae`.

Старые детекции попали в графики:

- перед новым прогоном можно очистить старый лог:

```bash
rm -f /tmp/trash_detections.jsonl
rm -rf /tmp/trash_detected
```

CUDA OOM при обучении на RTX 3060 Ti:

- использовать `imgsz=768`, `batch=8`;
- добавить `PYTORCH_ALLOC_CONF=expandable_segments:True`;
- не запускать Gazebo/RViz во время долгого обучения.

## Документация

- `docs/experiments.md` - запуск видеодемо, запись кадров, оценка прогона;
- `docs/datasets.md` - датасеты, mapping классов, TACO/Roboflow;
- `docs/diploma_status_and_plan.md` - статус диплома и следующий рабочий план;
- `models/trash/ASSET_SOURCES.md` - источники 3D-моделей и лицензии.
