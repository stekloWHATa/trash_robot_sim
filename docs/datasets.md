# Датасеты и обучение YOLOv8

## Целевая таксономия

Финальные классы описаны в `config/trash_classes.yaml`:

1. `cigarette_butt`
2. `plastic_bottle`
3. `glass_bottle`
4. `aluminum_can`
5. `plastic_bag`
6. `cardboard_box`
7. `paper_packaging`
8. `other_trash`

Классы исходных датасетов приводятся к этой таксономии через aliases. Широкие
материальные категории вроде `BIODEGRADABLE`, `Recyclable`, `PLASTIC`, `METAL`
и `GLASS` по умолчанию игнорируются, чтобы не размазывать модель по слишком
общим объектам.

## Источники

Уже лежат локально:

- `data/garbage_class3` - Roboflow `garbage-classification-3`, 6 широких классов;
- `data/garbage_segregation` - Roboflow `garbage-segregation-yyhof`, 44 класса;
- `data/merged` - старый объединенный набор на 48 классов;
- `data/runs/trash_finetune` - старый прогон YOLOv8n.

`data/merged` не считаем финальным датасетом для диплома: в нем много широких
или нерелевантных классов, нет нормального покрытия `cigarette_butt`, а старая
модель по нему не дает пригодного качества для видеодемо.

Нужно добавить перед финальным обучением:

- TACO official: COCO-like annotations, litter in the wild, источник для cigarette/bottle/can/carton/bag классов;
- отдельный Roboflow-набор с `cigarette_butt`, потому что в текущем `data/merged` этот класс нормально не представлен.

## Shortlist хороших источников

Дата поиска: 06.05.2026.

Цель shortlist - не просто набрать больше изображений, а закрыть конкретные
объекты демо: окурки, бутылки, банки, пакеты, коробки/бумажную упаковку. Старые
широкие классы `plastic`, `metal`, `glass`, `recyclable` оставляем только как
резерв или полностью игнорируем через `config/trash_classes.yaml`.

### Tier A - качать первыми

| Источник | Размер/формат | Лицензия | Что закрывает | Решение |
|---|---:|---|---|---|
| [TACO official Zenodo](https://zenodo.org/records/3587843) / [GitHub](https://github.com/pedropro/TACO) | 2.7 GB, COCO-like segmentation/bbox | CC BY 4.0 | `Cigarette`, `Clear plastic bottle`, `Drink can`, `Corrugated carton`, `Drink carton`, `Garbage bag`, `Glass bottle` и другие litter-классы | Обязательный базовый источник, конвертировать COCO -> YOLO |
| [Roboflow: Cigarette Butt Detection_Web](https://universe.roboflow.com/kimchidetector/cigarette-butt-detection_web-pox6p) | 2.2k images, Object Detection | CC BY 4.0 | `Cigarette_butt`, `Cigarette_Butt` | Основной источник окурков, маппить оба класса в `cigarette_butt` |
| [Roboflow: Cigarette Butt Detection](https://universe.roboflow.com/cigarette-butt-dr4wf/cigarette-butt-detection-qyi4u-ajbfw) | 2.4k images, Object Detection | MIT | `Cigarette_butt` | Второй источник окурков, использовать после визуального audit |
| [Roboflow: Waste-Detection](https://universe.roboflow.com/amulya-xc3vy/waste-detection-0momv-alnep) | 5.7k images, Object Detection | CC BY 4.0 | `can`, `Pet_Bottle`, `Plastic_Bag`, `Garbage_Bag`, `Paper_Bag`, `Glass` | Хороший источник бутылок/банок/пакетов, `Glass` как broad-class лучше не маппить в `glass_bottle` без проверки |
| [Roboflow: trash_detect](https://universe.roboflow.com/trashdetect-wamod/trash_detect-subnn) | 5.7k images, Object Detection, 28 classes | CC BY 4.0 | `bottle`, `can`, `cardboard`, `carton`, `glass bottle`, `plastic bag`, `plastic bottle`, `uht carton` | Хороший многоклассовый источник для целевой таксономии |

### Tier B - полезные, но проверить вручную перед merge

| Источник | Размер/формат | Лицензия | Что закрывает | Риск |
|---|---:|---|---|---|
| Roboflow: `Trash detection` by Soepkippen | 9.81k images, Object Detection | проверить в `Cite This Project` | `can`, `cardboard`, `plastic`, `cigarette_butt`, `paper_cup`, `pet_bottle` | Очень полезный набор, но broad-class `plastic` нельзя слепо маппить |
| Roboflow: `INSPIRE_Phase1_512` by VITOdronetemse | 1.3k images, Object Detection | проверить в `Cite This Project` | `Cigarette_butt`, `Glass_bottle`, `Metal_drinkcan`, `Paper_carton`, `Plastic_bag`, `Plastic_bottle` | Почти идеальные классы под диплом, но надо проверить лицензию и качество bbox |
| Roboflow: `YoloRecycling` | 10.9k images, Object Detection | проверить в `Cite This Project` | `bag_paper`, `bag_plastic`, `bottle_glass`, `bottle_plastic`, `can_metal`, `cardboard_paper` | Может быть больше похоже на сортировку/контейнеры, чем на мусор на полу |
| [MJU-Waste](https://www.mdpi.com/1424-8220/20/14/3816) | RGBD, segmentation, один класс `waste` | смотреть первичный источник | objectness/сегментация мусора | Не подходит для классификации 5+ классов, можно использовать только как binary waste pretrain |

### Почему эти источники лучше старого `data/merged`

- Есть отдельные object-detection наборы по окуркам, а не случайная broad-категория.
- Есть классы, похожие на реальные объекты демо: `Pet_Bottle`, `Drink can`,
  `Plastic_Bag`, `Corrugated carton`, `Paper_carton`.
- Можно выкинуть шумные общие классы до обучения, а не заставлять YOLO учить
  `PLASTIC`/`METAL`/`GLASS` как материалы.
- Roboflow Universe позволяет смотреть классы и лицензию в `Cite This Project`,
  а экспорт в YOLOv8 можно скачать ZIP-ом или кодом.

## Быстрый эксперимент: TACO-only

Да, TACO можно попробовать вообще без Roboflow. Это хороший первый baseline:
официальный источник, реальные сцены, COCO-like аннотации и понятная лицензия.
В дипломе это формулируется чище, чем смесь из нескольких Roboflow-проектов:
“модель обучена на открытом датасете TACO и протестирована в Gazebo/RViz”.

Ограничения TACO-only:

- датасет небольшой: около 1.5k images;
- классы сильно дисбалансны: `cigarette` и `clear_plastic_bottle` представлены
  хорошо, часть упаковки/пакетов/коробок заметно слабее;
- для цели `mAP50 >= 0.80` лучше сначала считать TACO-only baseline, а не
  обещать финальную цифру без аудита;
- если нужны стабильные 80-85% именно на окурках и мелком мусоре в видео, после
  TACO-only может понадобиться добавка отдельных cigarette-butt наборов.

Сборка только из TACO:

```bash
python3 scripts/train_yolo.py \
  --skip-download \
  --class-config config/trash_classes_mvp.yaml \
  --coco-json /path/to/TACO/data/annotations.json \
  --coco-images /path/to/TACO/data \
  --coco-output data/taco_yolo \
  --coco-split auto \
  --prepare-only

python3 scripts/train_yolo.py \
  --skip-download \
  --class-config config/trash_classes_mvp.yaml \
  --extra-dataset data/taco_yolo \
  --output-dir data/taco_only_detection_v1 \
  --prepare-only

python3 scripts/audit_dataset.py data/taco_only_detection_v1/data.yaml
python3 scripts/check_dataset_readiness.py data/taco_only_detection_v1/data.yaml
```

Обучение TACO-only baseline:

```bash
python3 scripts/train_yolo.py \
  --skip-download \
  --class-config config/trash_classes_mvp.yaml \
  --extra-dataset data/taco_yolo \
  --output-dir data/taco_only_detection_v1 \
  --epochs 120 \
  --batch 16 \
  --imgsz 960 \
  --base-model models/yolov8s.pt \
  --output-model models/yolov8s_trash.pt \
  --run-name taco_only_yolov8s_img960
```

Для мелких окурков лучше сразу планировать второй прогон с большим `imgsz`:

```bash
yolo detect train \
  model=yolov8s.pt \
  data=data/taco_only_detection_v1/data.yaml \
  epochs=120 \
  imgsz=960 \
  batch=16 \
  project=data/runs \
  name=taco_only_yolov8s_img960
```

Критерий принятия TACO-only:

- если `cigarette_butt`, `plastic_bottle`, `aluminum_can`, `plastic_bag`,
  `cardboard_box` имеют приемлемые precision/recall на validation и хорошо
  выглядят в demo video - оставляем TACO-only как основной дипломный набор;
- если проседают окурки или пакеты - фиксируем TACO-only как честный baseline и
  добавляем Roboflow cigarette-butt наборы отдельным экспериментом `TACO+CB`.

## Roboflow MVP v1

Собран локальный Roboflow-only датасет:

- путь: `data/merged_roboflow_mvp/data.yaml`;
- источники:
  - `data/raw/roboflow/cigarette_butt_kimchi_v4` - `Cigarette_butt`, MIT,
    https://universe.roboflow.com/kimchidetector/cigarette-butt-detection-qyi4u/dataset/4;
  - `data/raw/roboflow/trash_fyp` - can/cardboard/drink carton/plastic bag/plastic bottle, CC BY 4.0,
    https://universe.roboflow.com/fyp-bfx3h/yolov8-trash-detections/dataset/5;
  - `data/raw/roboflow/litterpicker` - Cardboard/Clear Plastic Bottle/Drink Can/Crisp Packet/Paper Cup, CC BY 4.0,
    https://universe.roboflow.com/litterpicker/litter-detection-dataset/dataset/13.
- сборка выполнялась через `config/trash_classes_mvp.yaml`, только 6 классов:
  `cigarette_butt`, `plastic_bottle`, `aluminum_can`, `plastic_bag`,
  `cardboard_box`, `paper_packaging`;
- старые локальные `garbage_class3`/`garbage_segregation` в этот набор не входят.

Команда сборки:

```bash
python3 scripts/train_yolo.py \
  --skip-download \
  --prepare-only \
  --only-extra-datasets \
  --class-config config/trash_classes_mvp.yaml \
  --output-dir data/merged_roboflow_mvp \
  --extra-dataset data/raw/roboflow/cigarette_butt_kimchi_v4 \
  --extra-dataset data/raw/roboflow/trash_fyp \
  --extra-dataset data/raw/roboflow/litterpicker
```

Результат audit:

- images: 11273;
- boxes: 27830;
- invalid labels: 0;
- empty labels: 3440;
- split: train 9700 images / 26628 bbox, valid 1184 images / 1056 bbox,
  test 389 images / 146 bbox.

Распределение классов:

| Class | BBox |
|---|---:|
| `paper_packaging` | 7942 |
| `aluminum_can` | 4362 |
| `plastic_bottle` | 4328 |
| `cigarette_butt` | 4128 |
| `plastic_bag` | 3594 |
| `cardboard_box` | 3476 |

`scripts/check_dataset_readiness.py data/merged_roboflow_mvp/data.yaml`
показывает `ready_for_yolov8s: True`. Главный риск этого набора - слабый
официальный test split: только 146 bbox, поэтому финальные дипломные метрики
лучше считать на отдельном validation/test или на Gazebo benchmark.

## План сборки `merged_detection_v1`

1. Скачать TACO official и конвертировать COCO/TACO:

```bash
python3 scripts/train_yolo.py \
  --skip-download \
  --class-config config/trash_classes_mvp.yaml \
  --coco-json /path/to/TACO/data/annotations.json \
  --coco-images /path/to/TACO/data \
  --coco-output data/taco_yolo \
  --coco-split auto \
  --prepare-only
```

2. Скачать Roboflow-наборы в YOLOv8-формате в отдельные папки, например:

```text
data/raw/cigarette_butt_web/
data/raw/cigarette_butt_detection/
data/raw/waste_detection/
data/raw/trash_detect/
```

Roboflow официально дает два варианта: ZIP или download code. Лицензию каждого
набора проверять в `Cite This Project` перед скачиванием. Если используется CLI:

```bash
roboflow download -f yolov8 -l data/raw/<target_dir> <workspace>/<project>/<version>
```

3. Собрать единый датасет, не трогая старый `data/merged`:

```bash
python3 scripts/train_yolo.py \
  --skip-download \
  --class-config config/trash_classes_mvp.yaml \
  --extra-dataset data/taco_yolo \
  --extra-dataset data/raw/cigarette_butt_web \
  --extra-dataset data/raw/cigarette_butt_detection \
  --extra-dataset data/raw/waste_detection \
  --extra-dataset data/raw/trash_detect \
  --output-dir data/merged_detection_v1 \
  --prepare-only
```

4. Проверить распределение классов:

```bash
python3 scripts/audit_dataset.py data/merged_detection_v1/data.yaml
python3 scripts/check_dataset_readiness.py data/merged_detection_v1/data.yaml
```

Целевой минимум перед долгим обучением:

- `cigarette_butt`: желательно 3000+ bbox, потому что объект мелкий;
- `plastic_bottle`, `aluminum_can`, `plastic_bag`, `cardboard_box`: 1000+ bbox;
- `glass_bottle`, `paper_packaging`: 500-1000+ bbox или объединить в более
  устойчивые классы для MVP;
- invalid labels: 0;
- val/test: не из тех же Roboflow-аугментаций, что train.

## План обучения на 80-85%

Реалистичная стратегия:

1. Не обучать финальную модель на 8 классах сразу, если данных мало. Для видео
   можно начать с 5-6 устойчивых классов:
   `cigarette_butt`, `plastic_bottle`, `aluminum_can`, `plastic_bag`,
   `cardboard_box`, `paper_packaging`.
2. Использовать `yolov8s` как основной кандидат; `yolov8n` оставить как быстрый
   baseline, `yolov8m` - если есть GPU-время.
3. Для окурков использовать `imgsz=960` или `1280`, сильнее сохранять мелкие
   bbox и проверить SAHI/tiling на inference.
4. Если удается запустить симуляцию без ручных действий, использовать
   `scripts/capture_ros_images.py` для автоматического сохранения кадров из
   `/rgbd/image/image` и `/detections_img`. Если автоматический запуск Gazebo в
   окружении недоступен, этот пункт остается optional и не блокирует обучение.
5. После каждого merge запускать audit и preview bbox, иначе можно получить
   красивую цифру на грязной разметке и плохую демку.

Для каждого нового источника фиксировать в отдельной таблице/заметке:

- ссылка на dataset/project;
- лицензия и условия атрибуции;
- исходные классы и mapping в `config/trash_classes.yaml`;
- число images и bbox по split;
- пригодность для мелкого мусора: bbox size, ракурсы, фон, motion blur.

## Подготовка нового набора

Без скачивания и без обучения:

```bash
python3 scripts/train_yolo.py \
  --skip-download \
  --class-config config/trash_classes_mvp.yaml \
  --prepare-only
```

Результат:

- `data/merged_v2/data.yaml`;
- `data/merged_v2/merge_report.yaml`;
- YOLO-структура `train|valid|test/images|labels`.

Если в исходном YOLO-файле встречается segmentation polygon формата
`class x1 y1 x2 y2 ...`, pipeline сворачивает polygon в bbox. Финальная модель
обучается в режиме object detection.

Добавление локально конвертированного TACO:

```bash
python3 scripts/train_yolo.py \
  --skip-download \
  --class-config config/trash_classes_mvp.yaml \
  --extra-dataset data/taco_yolo \
  --prepare-only
```

## COCO/TACO -> YOLO

```bash
python3 scripts/train_yolo.py \
  --skip-download \
  --class-config config/trash_classes_mvp.yaml \
  --coco-json /path/to/annotations.json \
  --coco-images /path/to/images \
  --coco-output data/taco_yolo \
  --coco-split auto \
  --prepare-only
```

Конвертер:

- читает `images`, `annotations`, `categories`;
- переводит COCO bbox `x,y,w,h` в YOLO `class xc yc w h`;
- применяет `config/trash_classes_mvp.yaml` по умолчанию; полную таксономию
  можно явно включить через `--class-config config/trash_classes.yaml`;
- пропускает unmapped классы;
- пишет отчет `<split>_coco_convert_report.yaml`; для `--coco-split auto`
  автоматически делает примерно 80/10/10 train/valid/test.

## Аудит

```bash
python3 scripts/audit_dataset.py data/merged_v2/data.yaml
```

Проверяется:

- количество images/labels/boxes;
- orphan images и orphan labels;
- empty labels;
- invalid class id / bbox;
- class distribution;
- min/max/mean/median bbox area.

Перед долгим обучением в `audit.md` не должно быть invalid labels. По
распределению классов нужно отдельно проверить `cigarette_butt`: если bbox мало,
надо добавить датасет или oversampling.
Дополнительно запускать:

```bash
python3 scripts/check_dataset_readiness.py data/merged_detection_v1/data.yaml
```

## Текущее качество старой модели

`data/runs/trash_finetune/results.csv`, epoch 100:

- precision: 0.5255;
- recall: 0.2916;
- mAP50: 0.3337;
- mAP50-95: 0.2335.

Это baseline, который нужно превзойти на `taco_only_detection_v1` и
`merged_detection_v1`.

## Финальное обучение

После аудита и добавления окурков:

```bash
python3 scripts/train_yolo.py \
  --skip-download \
  --class-config config/trash_classes_mvp.yaml \
  --extra-dataset data/taco_yolo \
  --output-dir data/merged_detection_v1 \
  --epochs 120 \
  --batch 16 \
  --imgsz 960 \
  --base-model models/yolov8s.pt \
  --output-model models/yolov8s_trash.pt \
  --run-name trash_yolov8s_img960
```

Результаты сравнивать с baseline:

- `metrics/precision(B)`;
- `metrics/recall(B)`;
- `metrics/mAP50(B)`;
- `metrics/mAP50-95(B)`;
- FPS/latency в `trash_detector`;
- качество локализации в Gazebo по ground truth.
