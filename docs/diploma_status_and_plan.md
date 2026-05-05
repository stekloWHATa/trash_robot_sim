# Ревизия дипломной работы и подробный план

Дата ревизии: 29.04.2026.

## 1. Что требуют документы

НИР, период 09.02.2026-18.04.2026:
- тема: исследование методов локализации и идентификации бытового мусора в рабочей сцене мобильного робота;
- нужно изучить методы детекции/сегментации, пространственную локализацию по глубине, существующие датасеты бытового мусора;
- практически нужно выбрать и адаптировать YOLOv8, сравнить с базовыми методами по точности и быстродействию.

Преддипломная практика, период 20.04.2026-16.05.2026:
- цель: реализовать и протестировать программный комплекс локализации и идентификации мусора в ROS 2/Gazebo;
- неделя 1, 19.04.2026-26.04.2026: YOLOv8-детекция и интеграция в ROS 2;
- неделя 2, 27.04.2026-03.05.2026: пространственная локализация объектов и Gazebo-интеграция;
- неделя 3, 04.05.2026-10.05.2026: эксперименты, сбор и оформление данных;
- неделя 4, 11.05.2026-16.05.2026: демонстрационные материалы, видео и выводы.

## 2. Что уже есть в проекте

Пакет ROS 2:
- `trash_robot_sim` собирается как `ament_cmake` пакет;
- `launch/gazebo.launch.py` запускает Gazebo Harmonic, робота, bridge, `navigator.py`, `map_builder.py`, `detector.py`;
- `launch/rviz.launch.py` и `rviz/robot.rviz` дают визуализацию карты, лидара, камер, пути и маркеров мусора;
- `worlds/trash_world.sdf`, `models/robot/only_robot.sdf`, `urdf/trash_robot.urdf.xacro` формируют симуляционную сцену и модель робота.

Навигация:
- `scripts/navigator.py` реализует A* по сетке, boustrophedon-покрытие области, pure-pursuit движение, обработку stuck-сценария и динамических препятствий по лидару;
- есть автоскан арены и ручные цели через `/scan_area` или `/goal_pose`;
- написаны сильные unit-тесты на координаты, инфляцию, A*, crash-сценарии и маршрут покрытия.

Карта:
- `scripts/map_builder.py` строит occupancy grid по LaserScan через log-odds;
- есть фильтр самодетекции лидара и пропуск сканов при высокой угловой скорости;
- детекция мусора из `map_builder.py` уже вынесена в отдельный YOLO-узел.

Детекция:
- `scripts/detector.py` загружает YOLOv8, публикует `/trash_markers`, `/trash_report`, `/detections_img`;
- для COCO-модели есть фильтр `COCO_TRASH_MAP`;
- для fine-tuned модели все классы считаются мусором;
- есть регистрация объектов с объединением повторных детекций, сохранение full/crop кадров в `/tmp/trash_detected`;
- есть преобразование bbox center + depth + odometry в мировые координаты.

YOLO и данные:
- `scripts/train_yolo.py` уже объединяет несколько Roboflow-датасетов, чинит `data.yaml`, дедуплицирует классы, перемаппит label id, собирает `data/merged` и запускает обучение YOLOv8n;
- локально есть `data/garbage_class3`, `data/garbage_segregation`, `data/merged`;
- локально есть `models/yolov8n.pt` и `models/yolov8n_trash.pt`;
- есть завершенный прогон `data/runs/trash_finetune`.

Текущий аудит `data/merged`:
- train: 22299 images / 22299 labels;
- valid: 3759 images / 3759 labels;
- test: 1871 images / 1871 labels;
- всего: 27929 изображений, 48 классов, 101989 bbox;
- invalid label id: 0;
- empty label files: 85;
- наиболее частые классы: `BIODEGRADABLE` 45407, `GLASS` 7809, `Recyclable` 6276, `CARDBOARD` 6268, `PLASTIC` 5945, `METAL` 5841, `PAPER` 4839, `Aluminum can` 4442, `Plastic bottle` 3057, `Plastic bag` 2131.

Текущий результат обучения `trash_finetune`:
- epochs: 100, imgsz: 640, batch: 16, optimizer: AdamW, device: `0`;
- длительность: 15297.8 s, примерно 4 ч 15 мин;
- итоговые метрики val: precision 0.5255, recall 0.2916, mAP50 0.3337, mAP50-95 0.2335;
- вывод: модель уже обучена, но качество слабое для дипломной демонстрации; главная проблема - смесь слишком широких классов, дисбаланс и отсутствие/недостаток ключевого класса `cigarette_butt`.

Тесты:
- добавлены `tests/test_train_yolo.py`;
- теперь `train_yolo.py` можно импортировать без установленного `ultralytics`, потому что импорт YOLO перенесен внутрь `train()`;
- тесты проверяют нормализацию Roboflow `data.yaml`, дедупликацию классов, перемаппинг label id, поддержку dict-формата `names`, создание пустых labels для неразмеченных изображений;
- полный локальный прогон: `python3 -m pytest tests -q` -> 49 passed.

Обновление следующего этапа:
- добавлен `config/trash_classes.yaml` с целевой таксономией на 8 классов;
- `scripts/train_yolo.py` теперь умеет применять class mapping, пропускать unmapped/ignored классы, сворачивать segmentation polygon в bbox, писать `data/merged_v2/merge_report.yaml`, конвертировать COCO/TACO в YOLO и подключать локальные наборы через `--extra-dataset`;
- добавлен `scripts/audit_dataset.py` для проверки YOLO-разметки перед обучением;
- `trash_detector` переведен на согласованные RGBD-топики по умолчанию, чтобы RGB и depth были из одной камеры;
- добавлены тесты на audit pipeline и геометрию детектора;
- README и `docs/datasets.md` обновлены под YOLOv8-архитектуру.

## 3. Ключевые проблемы, которые надо закрыть

1. Датасет не совпадает с целевыми объектами диплома.
   Сейчас есть общие классы мусора и контейнерные категории, но нет нормального покрытия окурков. Нужно явно добавить `cigarette_butt` и привести классы к целевой таксономии.

2. Классы надо укрупнить и нормализовать.
   Нельзя оставлять 48 разнородных классов как финальную модель. Для робота лучше 6-8 целевых классов:
   `cigarette_butt`, `plastic_bottle`, `glass_bottle`, `aluminum_can`, `plastic_bag`, `cardboard_box`, `paper_packaging`, `other_trash`.

3. Есть риск некорректной 3D-локализации.
   В `detector.py` RGB берется с `/camera/image`, а depth - с `/rgbd/image/depth_image`. Если камеры физически разные, bbox pixel center не обязан совпадать с depth pixel. Нужно либо использовать RGB и depth из одной RGBD-камеры, либо считать координаты пересечением луча NavCamera с плоскостью пола без depth, либо добавить depth именно для NavCamera.

4. Параметры ROS не доведены.
   `config/params.yaml` все еще содержит старые параметры цветовой детекции для `map_builder`, но не содержит явных параметров `trash_detector`: `model_path`, `conf_thresh`, `merge_dist`, `detect_rate`.

5. README устарел.
   README описывает цветовую детекцию и старую архитектуру, хотя фактически уже есть `detector.py` с YOLOv8.

6. Обучение не воспроизводится как эксперимент.
   Есть артефакты обучения, но нет единого отчета: версии датасетов, mapping классов, распределение классов, baseline COCO, baseline текущей модели, итоговая модель, таблица метрик, параметры запуска.

7. Не хватает тестов на геометрию детектора и валидацию датасета.
   Сейчас хорошо покрыт навигатор и начат train pipeline. Нужно добавить тесты для `_sample_depth`, `_pixel_to_world`, `_register`, data-audit CLI и smoke-тест launch/config.

## 4. Подходящие датасеты

Основная рекомендация:
- оставить текущие Roboflow `garbage-classification-3` и `garbage-segregation-yyhof` как источник общих классов;
- добавить официальный TACO как качественный источник litter-классов: cigarette, bottles, cans, cartons, paper/plastic bags;
- добавить отдельный Roboflow-датасет по `cigarette_butt`, потому что текущий локальный merged-набор этого класса не содержит.

Проверенные источники:
- TACO official Zenodo/GitHub: open dataset, trash/litter in the wild, COCO-like annotations, CC BY 4.0 на Zenodo.
  Links: https://zenodo.org/records/3587843, https://github.com/pedropro/TACO, https://arxiv.org/abs/2003.06975
- Roboflow `GARBAGE CLASSIFICATION 3`: 10464 images, YOLOv8 export, текущий источник `garbage_class3`.
  Link: https://universe.roboflow.com/material-identification/garbage-classification-3/dataset/2
- Roboflow search by `cigarette_butt`: есть несколько кандидатов, включая `Trash detection` 9.81k images с классами can/cardboard/plastic/cigarette_butt/paper_cup/pet_bottle, `INSPIRE_Phase1_512` 1.3k images с cigarette_butt/plastic_bag/plastic_bottle/metal_drinkcan/paper_carton и отдельные cigarette-butt-only наборы.
  Link: https://universe.roboflow.com/search?q=class%3Acigarette_butt

Критерии выбора датасета:
- формат object detection или instance segmentation, который можно конвертировать в YOLO bbox;
- лицензия CC BY 4.0 / Public Domain / явно допустимая для учебного проекта;
- есть классы именно мелкого мусора на полу/улице;
- есть `cigarette_butt`, `plastic_bottle`, `can`, `plastic_bag`, `cardboard` или близкие;
- нет сильного перекоса в один класс без возможности балансировки;
- изображения достаточно похожи на задачу робота: мусор на полу/земле, а не только мусорные баки или сортировочные ленты.

## 5. Подробный план работ

### Этап 1. Зафиксировать целевую таксономию

1. Создать файл `config/trash_classes.yaml`.
2. Утвердить финальные классы:
   `cigarette_butt`, `plastic_bottle`, `glass_bottle`, `aluminum_can`, `plastic_bag`, `cardboard_box`, `paper_packaging`, `other_trash`.
3. Описать mapping:
   - `Aluminum can`, `Tin`, `METAL`, `Drink can` -> `aluminum_can`, но не все `METAL`;
   - `Plastic bottle`, `PET_bottle`, `Clear plastic bottle`, `Milk bottle` -> `plastic_bottle`;
   - `Glass bottle`, часть `GLASS` -> `glass_bottle`;
   - `Plastic bag`, `Garbage bag`, `Zip plastic bag`, `Single-use carrier bag` -> `plastic_bag`;
   - `CARDBOARD`, `Corrugated carton`, `Postal packaging`, `Tetra pack`, `Other carton` -> `cardboard_box` или `paper_packaging`;
   - `Cigarette`, `Cigarette_butt`, `butt`, `cigarettes-ends` -> `cigarette_butt`.
4. Решить, что делать с слишком общими классами `BIODEGRADABLE`, `Recyclable`, `Non-Recyclable`: скорее исключить из обучения целевой модели или маппить в `other_trash` ограниченно.

### Этап 2. Переработать pipeline датасета

1. Расширить `DATASETS` в `scripts/train_yolo.py` или вынести список датасетов в YAML.
2. Добавить поддержку class mapping перед слиянием:
   - входной class name -> canonical class name;
   - классы без mapping пропускаются;
   - статистика пропущенных bbox сохраняется в отчет.
3. Добавить поддержку COCO/TACO:
   - скачать TACO;
   - сконвертировать COCO annotations в YOLO bbox;
   - применить mapping TACO -> canonical classes.
4. Добавить `scripts/audit_dataset.py`:
   - количество images/labels по split;
   - orphan images / orphan labels;
   - invalid class id;
   - empty labels;
   - class distribution;
   - min/max/median bbox area;
   - сохранение `data/merged/audit.json` и `data/merged/audit.md`.
5. Добавить генератор preview:
   - 20-50 случайных размеченных кадров;
   - overlay bbox/class/conf;
   - сохранить в `data/merged/preview/`.

### Этап 3. Собрать нормальный датасет для YOLOv8

1. Сформировать `data/merged_v2`.
2. Балансировать классы:
   - не давать `BIODEGRADABLE` или broad-классам доминировать;
   - целиться хотя бы в 1000+ bbox на основной класс, для `cigarette_butt` собрать максимум доступного;
   - если окурков мало, сделать oversampling/augmentation только для них.
3. Проверить leakage:
   - не допускать одинаковые изображения/аугментации в train и val/test;
   - Roboflow-generated файлы с похожими stem/hash проверять отдельно.
4. Сохранить frozen split:
   - `data/merged_v2/data.yaml`;
   - `data/merged_v2/class_mapping.yaml`;
   - `data/merged_v2/audit.md`.

### Этап 4. Обучение и сравнение YOLOv8

1. Baseline 1: `yolov8n.pt` COCO без fine-tune в симуляции.
2. Baseline 2: текущий `models/yolov8n_trash.pt`.
3. Candidate A: `yolov8n` на `merged_v2`, 80-120 epochs.
4. Candidate B: `yolov8s` на `merged_v2`, если есть GPU-время.
5. Для каждого прогона сохранять:
   - `args.yaml`;
   - `results.csv`;
   - confusion matrix;
   - PR/F1/P/R curves;
   - `best.pt`, `last.pt`;
   - short markdown report.
6. Целевые критерии:
   - mAP50 выше текущих 0.3337;
   - recall выше текущих 0.2916;
   - latency на машине диплома: FPS >= 4-5 для ROS-узла;
   - в Gazebo робот стабильно находит целевые объекты с приемлемыми false positives.

### Этап 5. Исправить ROS-интеграцию детектора

1. В `config/params.yaml` добавить блок:
   - `trash_detector.ros__parameters.model_path`;
   - `conf_thresh`;
   - `merge_dist`;
   - `detect_rate`;
   - `camera_mode`: `rgbd` или `nav_ground_plane`.
2. Исправить default model:
   - либо запускать `models/yolov8n_trash.pt`;
   - либо копировать best model в `models/yolov8n.pt` осознанно и документировать.
3. Решить проблему RGB/depth:
   - вариант A: использовать `/rgbd/image/image`, `/rgbd/image/depth_image`, `/rgbd/image/camera_info`;
   - вариант B: использовать `/camera/image` и считать пересечение луча с плоскостью пола по известной позе камеры;
   - вариант C: добавить depth к NavCamera в SDF.
4. Покрыть тестами:
   - `_sample_depth` с NaN/Inf/нулевыми значениями;
   - `_pixel_to_world` на простых известных позах;
   - `_register` на merge/new object;
   - mode detection COCO/fine-tuned.
5. Добавить логирование FPS/latency:
   - inference ms;
   - detections per frame;
   - skipped frames;
   - current model path.

### Этап 6. Эксперименты в Gazebo

1. Сформировать 3-5 сценариев:
   - простая сцена: 5 крупных объектов;
   - средняя сцена: 10-15 объектов разных классов;
   - сложная сцена: мелкие окурки + пакеты + банки рядом с препятствиями;
   - false-positive сцена: похожие, но нецелевые объекты;
   - полный прогон арены.
2. Добавить ground truth файл для объектов сцены:
   - class;
   - world x/y;
   - размер;
   - id модели Gazebo.
3. Написать `scripts/evaluate_sim_run.py`:
   - читает `/trash_report` или сохраненный лог;
   - матчинг детекций к ground truth по distance threshold;
   - precision/recall/F1 по объектам;
   - localization error mean/median/max;
   - latency/FPS.
4. Сохранить таблицы:
   - baseline COCO;
   - current trash_finetune;
   - final model;
   - effect of depth/floor-plane localization.
5. Снять видео:
   - Gazebo view;
   - RViz markers/map/path;
   - `/detections_img`.

### Этап 7. Документация и отчет

1. Обновить README под фактическую архитектуру YOLOv8.
2. Добавить раздел `docs/experiments.md`.
3. Добавить раздел `docs/datasets.md`:
   - источники;
   - лицензии;
   - mapping классов;
   - статистика split;
   - почему выбран YOLOv8.
4. Для диплома подготовить:
   - постановку задачи;
   - обзор методов;
   - описание ROS 2/Gazebo комплекса;
   - описание датасета и обучения;
   - метрики детекции и локализации;
   - выводы и ограничения.

## 6. Идеальный промпт для следующего прохода Codex

Продолжи работу в репозитории `/home/steklowhata/ros2_ws/src/trash_robot_sim`.

Новый фокус диплома: убедительная **видеодемонстрация детекции и пространственной локализации бытового мусора мобильным роботом**. Главный артефакт - короткий scripted demo run: робот едет/поворачивается в заранее подготовленной сцене, камера видит реалистичные объекты мусора, YOLOv8 рисует bbox и классы, система вычисляет координаты объектов и отображает их на карте/RViz. Сейчас не тратим время на сложный автономный объезд препятствий, покрытие всей арены и поведение уборщика. Навигация (`navigator.py`, A*, boustrophedon, obstacle avoidance) остается как будущая доп. фича, но в ближайшем этапе ее не развиваем.

Текущий статус:
- есть ROS 2/Gazebo пакет `trash_robot_sim`;
- есть `detector.py` с YOLOv8, RGBD-локализацией, `/detections_img`, `/trash_markers`, `/trash_report`;
- есть `map_builder.py` с occupancy grid по лидару, его можно использовать только как карту/подложку для маркеров;
- есть `navigator.py`, но на этом этапе он не является приоритетом;
- есть `launch/detection_demo.launch.py`, `worlds/detection_demo_world.sdf` и `scripts/scripted_motion.py` для video-first MVP без запуска навигатора;
- есть локальные procedural mesh-модели мусора в `models/trash/...` и `config/trash_ground_truth.yaml` для оценки локализации;
- есть `scripts/train_yolo.py`, `scripts/audit_dataset.py`, `config/trash_classes.yaml`, `docs/datasets.md`;
- есть `scripts/evaluate_detection_run.py`, который считает TP/FP/FN, F1, ошибку локализации и latency по JSONL-логу детектора;
- целевые классы: `cigarette_butt`, `plastic_bottle`, `glass_bottle`, `aluminum_can`, `plastic_bag`, `cardboard_box`, `paper_packaging`, `other_trash`;
- текущая старая модель `data/runs/trash_finetune`: precision 0.5255, recall 0.2916, mAP50 0.3337, mAP50-95 0.2335, качество недостаточно;
- быстрые тесты проходят: `python3 -m pytest tests -q` -> 54 passed.

Цель следующего этапа:
- собрать демонстрационный стенд “детекция + локализация” без зависимости от автономного объезда;
- подготовить воспроизводимый video-first сценарий: заранее расставленный мусор, понятный маршрут/teleop/scripted motion, RViz/Gazebo ракурсы, сохранение кадров и логов;
- заменить/добавить реалистичные mesh-модели мусора в Gazebo;
- собрать новый качественный датасет и training pipeline с целью `mAP50 >= 0.80`, желательно `0.85`, на validation/test и отдельном симуляционном benchmark;
- получить понятные экспериментальные метрики из реальных логов/симуляционного ground truth: class precision/recall/mAP, object-level F1 в Gazebo, средняя/медианная ошибка локализации в метрах, FPS/latency детектора.

Сделай следующий этап полностью:
1. Сначала проверь `git status`, не затирай чужие изменения и не удаляй `data/`/`models/`.
2. Перестрой план запуска под MVP детекции:
   - добавь отдельный launch, например `launch/detection_demo.launch.py`, который запускает Gazebo, robot_state_publisher, bridge, `map_builder` и `trash_detector`, но не требует `navigator`;
   - сделай параметры, позволяющие запускать робота статично, вручную через teleop или по простому scripted path без объезда препятствий;
   - добавь удобный способ записи демо: сохранение `/detections_img`, логов детектора, `/trash_report`, RViz/Gazebo инструкции для OBS;
   - README должен объяснять именно видеодемонстрацию детекции/локализации.
3. Улучши мусор в симуляции:
   - найди или подготовь реалистичные mesh-модели для `cigarette_butt`, plastic bottle, glass bottle, aluminum can, plastic bag, cardboard box/paper packaging;
   - работа учебная/некоммерческая, поэтому можно использовать не только CC0/CC BY, но и ассеты с лицензиями `CC BY-NC`, `Free for non-commercial use`, `Editorial/educational use`, BlenderKit/Sketchfab/Poly Haven/CGTrader/иные источники, если конкретная карточка модели явно разрешает такое использование;
   - Blender как программа не дает прав на любые чужие модели: для каждого `.blend/.obj/.fbx/.glb` нужно сохранить ссылку на источник, автора, тип лицензии и условия атрибуции;
   - не используй пиратские сливы, ripped game assets, модели без понятного автора/лицензии или файлы, где запрет на скачивание/переиспользование очевиден;
   - добавь модели в `models/trash/...`, подключи их в SDF через `<mesh><uri>...`;
   - сделай нормальные scale, visual, collision, pose, material/texture;
   - добавь `config/trash_ground_truth.yaml` с class/id/world pose для оценки локализации.
4. Пересобери датасетный план:
   - старый `data/merged` не считать финальным;
   - подобрать датасеты под реальные целевые классы, особенно `cigarette_butt`;
   - приоритетные источники: официальный TACO, Roboflow cigarette_butt datasets, Roboflow/Universe datasets с классами bottle/can/cardboard/plastic bag/glass bottle/paper packaging;
   - обязательно фиксировать лицензию, ссылку, классы, число изображений, число bbox и пригодность к задаче;
   - использовать `config/trash_classes.yaml`, `scripts/train_yolo.py`, `scripts/audit_dataset.py`;
   - если источник дает segmentation polygon, конвертировать в bbox для YOLO detect;
   - собрать `data/merged_detection_v1` или `data/merged_v3`, не ломая старый `data/merged`.
5. Доведи pipeline обучения до реалистичной цели 80-85%:
   - провести audit распределения классов;
   - сбалансировать классы, особенно окурки и мелкие объекты;
   - добавить augmentation под мелкий мусор: mosaic/copy-paste, random scale, blur/noise, brightness, perspective;
   - рассмотреть `yolov8s` или `yolov8m`, если `yolov8n` не дотягивает;
   - для мелких окурков рассмотреть tiling/SAHI inference или обучение с большим `imgsz`;
   - не запускать многочасовое обучение без явного решения, но подготовить команду, конфиг и expected outputs.
6. Улучши `detector.py` именно под локализацию:
   - убедись, что RGB и depth берутся из одной RGBD-камеры;
   - добавь стабильный формат результата: class, confidence, bbox, depth, world x/y, timestamp;
   - сохраняй экспериментальный лог в CSV/JSONL;
   - проверь `/trash_markers`, `/trash_report`, `/detections_img`;
   - добавь параметры confidence per class, min/max depth, merge distance, save crops.
7. Добавь оценку локализации и детекции в симуляции:
   - написать или доработать `scripts/evaluate_detection_run.py`;
   - вход: ground truth мусора из `config/trash_ground_truth.yaml` и лог детектора;
   - выход: TP/FP/FN по классам, precision/recall/F1, localization error mean/median/max, FPS/latency;
   - сохранить отчет в `data/eval/...`.
   - если нужны графики для отчета, генерируй их только из логов/ground truth или помечай как demo/synthetic illustration, не выдавай иллюстративные данные за реальные измерения.
8. Добавь тесты:
   - тесты class mapping/audit/dataset conversion;
   - тесты geometry/depth/world projection;
   - тесты evaluator matching ground truth <-> detections;
   - smoke-тест launch/config, если можно без запуска Gazebo.
9. Обнови документацию:
   - README: быстрый запуск detection demo;
   - `docs/datasets.md`: выбранные датасеты, лицензии, статистика, почему старые данные слабые;
   - `docs/experiments.md`: как запускать видеодемо, какие окна/топики записывать, какие метрики собирать, критерии готовности;
   - `docs/diploma_status_and_plan.md`: актуальный статус и следующий шаг.
10. Запусти быстрые тесты и в финальном ответе кратко дай:
   - что изменено;
   - какие файлы тронуты;
   - какие тесты прошли;
   - что осталось перед долгим GPU-обучением и финальной демонстрацией.

Ограничения:
- главный приоритет: YOLOv8 detection + RGBD/world localization + RViz/map markers;
- obstacle avoidance, A*, boustrophedon, full autonomous coverage пока не развивать;
- не запускать многочасовое обучение без явного решения;
- не фабриковать метрики и графики: для защиты можно делать красивую визуализацию, scripted demo и synthetic illustrations, но реальные численные результаты должны быть из логов/валидации/ground truth;
- не скачивать гигабайты без предупреждения;
- не удалять `data/` и `models/`;
- можно использовать модели, разрешенные для учебного/некоммерческого применения, но не использовать пиратские/неясно лицензированные 3D-модели;
- не делать unrelated refactor;
- использовать существующие стили проекта;
- все изменения проверять тестами.

Полезные легальные источники для поиска:
- TACO official: https://zenodo.org/records/3587843, https://github.com/pedropro/TACO
- Roboflow Universe search и datasets: проверять license в Cite This Project;
- Poly Haven: CC0 3D assets/textures/HDRI;
- BlenderKit: бесплатные/Full Plan assets, проверять лицензию конкретного asset и условия использования;
- Sketchfab: использовать Downloadable assets с CC0/CC BY/CC BY-NC/подходящей учебной лицензией, сохранять attribution и ссылку.
