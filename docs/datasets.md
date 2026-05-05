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

Для каждого нового источника фиксировать в отдельной таблице/заметке:

- ссылка на dataset/project;
- лицензия и условия атрибуции;
- исходные классы и mapping в `config/trash_classes.yaml`;
- число images и bbox по split;
- пригодность для мелкого мусора: bbox size, ракурсы, фон, motion blur.

## Подготовка нового набора

Без скачивания и без обучения:

```bash
python3 scripts/train_yolo.py --skip-download --prepare-only
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
  --extra-dataset data/taco_yolo \
  --prepare-only
```

## COCO/TACO -> YOLO

```bash
python3 scripts/train_yolo.py \
  --coco-json /path/to/annotations.json \
  --coco-images /path/to/images \
  --coco-output data/taco_yolo \
  --coco-split train \
  --prepare-only
```

Конвертер:

- читает `images`, `annotations`, `categories`;
- переводит COCO bbox `x,y,w,h` в YOLO `class xc yc w h`;
- применяет `config/trash_classes.yaml`;
- пропускает unmapped классы;
- пишет отчет `train_coco_convert_report.yaml`.

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

## Текущее качество старой модели

`data/runs/trash_finetune/results.csv`, epoch 100:

- precision: 0.5255;
- recall: 0.2916;
- mAP50: 0.3337;
- mAP50-95: 0.2335.

Это baseline, который нужно превзойти на `merged_v2`.

## Финальное обучение

После аудита и добавления окурков:

```bash
python3 scripts/train_yolo.py \
  --skip-download \
  --extra-dataset data/taco_yolo \
  --epochs 100 \
  --batch 16
```

Результаты сравнивать с baseline:

- `metrics/precision(B)`;
- `metrics/recall(B)`;
- `metrics/mAP50(B)`;
- `metrics/mAP50-95(B)`;
- FPS/latency в `trash_detector`;
- качество локализации в Gazebo по ground truth.
