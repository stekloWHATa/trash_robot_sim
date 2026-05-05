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

Логи:

- `/tmp/trash_detections.jsonl` - каждая строка содержит class, confidence,
  bbox, depth, world x/y, robot pose, latency.
- `/tmp/trash_detected` - полный кадр и crop для новых объектов.
- `config/trash_ground_truth.yaml` - эталонные позиции мусора в demo world.

## Оценка прогона

После остановки демо:

```bash
python3 scripts/evaluate_detection_run.py \
  --ground-truth config/trash_ground_truth.yaml \
  --detections /tmp/trash_detections.jsonl
```

По умолчанию отчет сохраняется в `data/eval/<timestamp>/report.json` и
`report.md`.

Скрипт считает:

- TP/FP/FN по классам;
- precision/recall/F1 по объектам в Gazebo;
- mean/median/max ошибку локализации в метрах;
- среднюю/медианную/max latency YOLO-инференса;
- оценку FPS по timestamp из лога.

Числа из этого отчета можно использовать в дипломе как результаты конкретного
sim-to-demo прогона. Иллюстративные графики допустимы только если явно
подписаны как synthetic/demo illustration, а не как реальные измерения.

## Критерии готовности видео

- В сцене видны минимум 5-6 типов мусора из целевой таксономии.
- `/detections_img` показывает bbox без сильных скачков и пропусков.
- `/trash_markers` появляются рядом с физическими объектами на карте.
- JSONL-лог не пустой, evaluator находит хотя бы часть ground truth объектов.
- Видео можно повторить одной командой `detection_demo.launch.py`.

## Следующий шаг после демо

Перед финальной защитой нужно заменить временную/старую модель на обученную
`best.pt`, собрать новый датасет с нормальными окурками и мелким мусором,
прогнать long training на GPU и повторить evaluator на том же demo world.
