# Эксперименты и видеодемонстрация

## Цель демо

Показать минимально убедительную систему: робот едет по подготовленной сцене,
RGBD-камера видит бытовой мусор, YOLOv8 рисует bbox, узел `trash_detector`
переводит detections в мировые координаты и публикует маркеры в RViz.

Автономный объезд препятствий, A* и полное покрытие арены в этом сценарии не
являются критичными. Для видео используется простой scripted motion или ручной
teleop.

## Запуск

```bash
cd ~/ros2_ws
colcon build --packages-select trash_robot_sim
source install/setup.bash
ros2 launch trash_robot_sim detection_demo.launch.py scripted_motion:=true
```

Второй терминал:

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch trash_robot_sim rviz.launch.py
```

Статичный режим для настройки ракурсов:

```bash
ros2 launch trash_robot_sim detection_demo.launch.py scripted_motion:=false
```

## Что записывать

- Gazebo viewport: робот, камера и реалистичные mesh-объекты мусора.
- RViz: `/map`, `/trash_markers`, TF, при необходимости `/detections_img`.
- Окно с `/detections_img`: bbox, класс и confidence на RGB-кадре.
- Терминал или сохраненный `/trash_report`: список найденных объектов.

## Сцена и маршрут

Текущий `detection_demo_world` сделан под видеодемонстрацию:

- рабочая плоскость расширена до 24 x 14 м;
- декоративные стены удалены, чтобы не ломать кадр Gazebo;
- мусор разложен вдоль зигзагообразной линии движения;
- в сцене по одному уникальному объекту MVP-классов:
  `cigarette_butt`, `aluminum_can`, `plastic_bottle`, `plastic_bag`,
  `cardboard_box`, `paper_packaging`;
- `scripts/scripted_motion.py` ведет робота по waypoint-маршруту через `/odom`,
  а не по грубым таймерам.

Маршрут по точкам:

```text
(0.0, -2.0) -> (1.8, -1.4) -> (3.6, -3.0) -> (5.4, -1.4)
             -> (7.2, -3.0) -> (9.0, -1.4) -> (10.8, -3.0)
```

Ground truth для оценки локализации лежит в
`config/trash_ground_truth.yaml`.

Логи:

- `/tmp/trash_detections.jsonl` - каждая строка содержит class, confidence,
  bbox, depth, world x/y, robot pose, latency и `source`;
- `/tmp/trash_detected` - полный кадр, crop, карточка `*_card.jpg` с классом,
  confidence, bbox, depth, world x/y, pose робота, latency и JSON-метаданные
  `*_meta.json` для новых объектов.
- `config/trash_ground_truth.yaml` - эталонные позиции мусора в demo world.

Автоматические кадры без ручных скриншотов:

Одна команда, которая сама запускает demo launch, сохраняет кадры и завершает
процесс:

```bash
source ~/ros2_ws/install/setup.bash
ros2 run trash_robot_sim run_demo_capture.py \
  --output-dir /tmp/trash_demo_capture \
  --max-frames 30 \
  --every-n 5
```

Если демо уже запущено отдельно, можно сохранить только image topics:

```bash
source ~/ros2_ws/install/setup.bash
ros2 run trash_robot_sim capture_ros_images.py \
  --topic /rgbd/image/image \
  --topic /detections_img \
  --output-dir /tmp/trash_demo_capture \
  --max-frames 30 \
  --every-n 5
```

Скрипт сохраняет JPEG-кадры из ROS image topics в
`/tmp/trash_demo_capture/rgbd__image__image/` и
`/tmp/trash_demo_capture/detections_img/`. Это не ручной screenshot: кадры
берутся напрямую из ROS-топиков.

## Оценка прогона

После остановки демо:

```bash
python3 scripts/evaluate_detection_run.py \
  --ground-truth config/trash_ground_truth.yaml \
  --detections /tmp/trash_detections.jsonl
```

По умолчанию отчет сохраняется в `data/eval/<timestamp>/report.json` и
`report.md`.

Отдельная папка с графиками/таблицами для диплома:

```bash
python3 scripts/generate_demo_report_assets.py \
  --ground-truth config/trash_ground_truth.yaml \
  --detections /tmp/trash_detections.jsonl \
  --output-dir reports/detection_demo/latest
```

Скрипт пишет `summary.md`, `summary.json`, CSV-таблицу и SVG-графики:
распределение детекций по классам, confidence/depth по классам, scatter
локализации в мировых координатах, latency histogram и timeline детекций.
Это реальные графики по JSONL-логу конкретного прогона.

Скрипт считает:

- TP/FP/FN по классам;
- precision/recall/F1 по объектам в Gazebo;
- mean/median/max ошибку локализации в метрах;
- среднюю/медианную/max latency YOLO-инференса;
- оценку FPS по timestamp из лога.

Числа из этого отчета можно использовать в дипломе как результаты конкретного
sim-to-demo прогона. Иллюстративные графики допустимы только если явно
подписаны как synthetic/demo illustration, а не как реальные измерения.

Важно: для красивого видеодемо включен `demo_ground_truth_assist`, который
стабильно рисует и публикует объекты подготовленной Gazebo-сцены, если YOLO
из-за sim-to-real gap пропускает mesh. Такие строки в JSONL имеют
`source=demo_ground_truth_assist`; для честных метрик YOLO этот режим нужно
выключить в `config/params.yaml` и отдельно валидировать `models/yolov8s_trash.pt`.

## Критерии готовности видео

- В сцене видны минимум 5-6 типов мусора из целевой таксономии.
- `/detections_img` показывает bbox без сильных скачков и пропусков.
- `/trash_markers` появляются рядом с физическими объектами на карте.
- JSONL-лог не пустой, evaluator находит хотя бы часть ground truth объектов.
- Видео можно повторить одной командой `detection_demo.launch.py`.

## Следующий шаг после демо

Перед финальной защитой нужно заменить временную/старую модель на обученную
`models/yolov8s_trash.pt`, собрать новый датасет с нормальными окурками и
мелким мусором, прогнать long training на GPU и повторить evaluator на том же
demo world.

## YOLOv8s Training Candidate

Текущий основной датасет для следующего GPU-прогона:

- `data/merged_roboflow_mvp/data.yaml`;
- 11273 images, 27830 bbox, 6 MVP-классов;
- readiness: `Ready for yolov8s/imgsz=960 training`;
- классы: `cigarette_butt`, `plastic_bottle`, `aluminum_can`, `plastic_bag`,
  `cardboard_box`, `paper_packaging`.

Команда долгого обучения, когда есть GPU-время:

```bash
python3 scripts/train_yolo.py \
  --skip-download \
  --only-extra-datasets \
  --class-config config/trash_classes_mvp.yaml \
  --extra-dataset data/raw/roboflow/cigarette_butt_kimchi_v4 \
  --extra-dataset data/raw/roboflow/trash_fyp \
  --extra-dataset data/raw/roboflow/litterpicker \
  --output-dir data/merged_roboflow_mvp \
  --epochs 120 \
  --batch 16 \
  --imgsz 960 \
  --base-model models/yolov8s.pt \
  --output-model models/yolov8s_trash.pt \
  --run-name roboflow_mvp_yolov8s_img960
```

После обучения проверить:

```bash
yolo detect val \
  model=models/yolov8s_trash.pt \
  data=data/merged_roboflow_mvp/data.yaml \
  imgsz=960
```

Для дипломных метрик не полагаться только на Roboflow test split: в нем сейчас
мало bbox. Нужен отдельный контрольный прогон в Gazebo через
`scripts/evaluate_detection_run.py`.
