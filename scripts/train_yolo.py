#!/usr/bin/env python3
"""
train_yolo.py — дообучение YOLOv8n на TACO (Trash Annotations in Context).

TACO — специализированный датасет мусора: ~1500 изображений, 60 классов мусора,
сгруппированных в 28 суперкатегорий (bottle, can, carton, cigarette, etc.)

Использование:
  # Установить зависимости (один раз):
  pip install ultralytics roboflow

  # Запустить обучение:
  python3 scripts/train_yolo.py

  # После завершения (~2-4 часа на GPU, ~12ч на CPU):
  # Модель сохраняется в models/yolov8n_trash.pt
  # Обновите detector.py параметр model_path или скопируйте как yolov8n.pt

Датасет загружается с Roboflow (TACO в формате YOLO):
  https://universe.roboflow.com/material-identification/taco-trash-annotations-in-context

Маппинг TACO-классов → наши категории задаётся в TACO_CATEGORY_MAP ниже.
"""

import os
import sys
import shutil
import argparse

# ── Проверка зависимостей ─────────────────────────────────────────────────── #
try:
    from ultralytics import YOLO
except ImportError:
    print('[ERROR] ultralytics не установлен. Запустите: pip install ultralytics')
    sys.exit(1)

# ── Пути ─────────────────────────────────────────────────────────────────── #
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR  = os.path.dirname(SCRIPT_DIR)
MODELS_DIR   = os.path.join(PACKAGE_DIR, 'models')
BASE_MODEL   = os.path.join(MODELS_DIR, 'yolov8n.pt')
OUTPUT_MODEL = os.path.join(MODELS_DIR, 'yolov8n_trash.pt')
DATA_DIR     = os.path.join(PACKAGE_DIR, 'data', 'taco_yolo')
DATA_YAML    = os.path.join(DATA_DIR, 'data.yaml')

# ── Параметры обучения ────────────────────────────────────────────────────── #
EPOCHS      = 50       # 50 хватает для fine-tune; 100 дадут чуть лучше
IMGSZ       = 640
BATCH       = 8        # уменьшить до 4 если OOM на GPU
PATIENCE    = 10       # early stopping
DEVICE      = '0'      # GPU (NVIDIA). Если ошибка — поменяй на 'cpu'

# ── Маппинг TACO 28 суперкатегорий → наши категории ─────────────────────── #
# TACO supers: Bottle, Can, Carton, Cup, Lid, Other plastic, Paper, Plastic bag,
#              Rope/Strings, Scrap metal, Shoe, Styrofoam, Unlabeled litter,
#              Cigarette, Paper bag, Straw, Plastic film, Pop tab,
#              Broken glass, Food waste, Glass jar, Battery, Blister pack,
#              Foam sponge, Plastic container, Squeezable tube, Wrapping foil, Rubber glove
TACO_CATEGORY_MAP = {
    # Точные имена классов из TACO v15 на Roboflow (59 классов)
    'Aerosol':                    'misc_object',
    'Aluminium blister pack':     'hygiene',
    'Aluminium foil':             'misc_object',
    'Battery':                    'electronics',
    'Broken glass':               'glass_bottle',
    'Carded blister pack':        'hygiene',
    'Cigarette':                  'cigarette',
    'Clear plastic bottle':       'plastic_bottle',
    'Corrugated carton':          'cardboard_paper',
    'Crisp packet':               'misc_object',
    'Disposable food container':  'misc_object',
    'Disposable plastic cup':     'can_cup',
    'Drink can':                  'can_cup',
    'Drink carton':               'cardboard_paper',
    'Egg carton':                 'cardboard_paper',
    'Foam cup':                   'can_cup',
    'Foam food container':        'misc_object',
    'Food Can':                   'can_cup',
    'Food waste':                 'organic_waste',
    'Garbage bag':                'misc_object',
    'Glass bottle':               'glass_bottle',
    'Glass cup':                  'can_cup',
    'Glass jar':                  'glass_bottle',
    'Magazine paper':             'cardboard_paper',
    'Meal carton':                'cardboard_paper',
    'Metal bottle cap':           'misc_object',
    'Metal lid':                  'misc_object',
    'Normal paper':               'cardboard_paper',
    'Other carton':               'cardboard_paper',
    'Other plastic bottle':       'plastic_bottle',
    'Other plastic container':    'plastic_bottle',
    'Other plastic cup':          'can_cup',
    'Other plastic wrapper':      'misc_object',
    'Other plastic':              'misc_object',
    'Paper bag':                  'cardboard_paper',
    'Paper cup':                  'can_cup',
    'Paper straw':                'misc_object',
    'Pizza box':                  'cardboard_paper',
    'Plastic bottle cap':         'misc_object',
    'Plastic film':               'misc_object',
    'Plastic glooves':            'hygiene',
    'Plastic lid':                'misc_object',
    'Plastic straw':              'misc_object',
    'Plastic utensils':           'misc_object',
    'Polypropylene bag':          'misc_object',
    'Pop tab':                    'can_cup',
    'Rope - strings':             'misc_object',
    'Scrap metal':                'misc_object',
    'Shoe':                       'misc_object',
    'Single-use carrier bag':     'misc_object',
    'Six pack rings':             'misc_object',
    'Spread tub':                 'misc_object',
    'Squeezable tube':            'hygiene',
    'Styrofoam piece':            'misc_object',
    'Tissues':                    'hygiene',
    'Toilet tube':                'cardboard_paper',
    'Tupperware':                 'plastic_bottle',
    'Unlabeled litter':           'misc_object',
    'Wrapping paper':             'misc_object',
}

OUR_CLASSES = sorted(set(TACO_CATEGORY_MAP.values()))


def download_taco_roboflow(api_key: str, dest: str):
    """Скачивает TACO с Roboflow в формате YOLOv8."""
    import time
    import zipfile
    try:
        from roboflow import Roboflow
    except ImportError:
        print('[ERROR] roboflow не установлен. Запустите: pip install roboflow')
        sys.exit(1)

    print('[INFO] Загрузка TACO с Roboflow...')
    rf = Roboflow(api_key=api_key)
    project = rf.workspace('material-identification').project(
        'taco-trash-annotations-in-context')
    version = project.version(15)

    # Триггерим генерацию экспорта
    print('[INFO] Генерация экспорта YOLOv8 (асинхронно)...')
    version.export('yolov8')

    # Скачиваем с retry — экспорт генерируется 10-30 секунд
    zip_path = os.path.join(dest, 'roboflow.zip')
    for attempt in range(1, 13):
        # Удаляем битый файл если есть
        if os.path.isfile(zip_path):
            os.remove(zip_path)
        print(f'[INFO] Попытка скачивания {attempt}/12...')
        try:
            dataset = version.download('yolov8', location=dest, overwrite=True)
            # Проверяем что zip валидный
            if os.path.isfile(zip_path):
                try:
                    with zipfile.ZipFile(zip_path):
                        pass
                except zipfile.BadZipFile:
                    print(f'[WARN] Экспорт ещё не готов, ждём 15с...')
                    time.sleep(15)
                    continue
            print(f'[INFO] Датасет сохранён: {dataset.location}')
            return dataset.location
        except Exception as e:
            if 'BadZipFile' in str(type(e).__name__) or 'not a zip' in str(e).lower():
                print(f'[WARN] Экспорт ещё не готов, ждём 15с...')
                time.sleep(15)
            else:
                print(f'[ERROR] {e}')
                sys.exit(1)

    print('[ERROR] Экспорт так и не сгенерировался за 3 минуты.')
    print('[INFO] Попробуйте скачать вручную:')
    print('  1. Откройте https://universe.roboflow.com/material-identification/taco-trash-annotations-in-context/15')
    print('  2. Export → YOLOv8 → download zip')
    print(f'  3. Распакуйте в {dest}')
    print(f'  4. Запустите: python3 train_yolo.py --skip-download')
    sys.exit(1)


def make_data_yaml(data_dir: str) -> str:
    """Создаёт data.yaml для обучения с нашими классами."""
    yaml_path = os.path.join(data_dir, 'data.yaml')
    class_names = OUR_CLASSES
    content = f"""# TACO → trash_robot_sim classes
path: {data_dir}
train: train/images
val:   valid/images
test:  test/images

nc: {len(class_names)}
names: {class_names}
"""
    with open(yaml_path, 'w') as f:
        f.write(content)
    print(f'[INFO] data.yaml: {yaml_path}')
    return yaml_path


def remap_labels(data_dir: str, taco_class_file: str):
    """
    Перемаппирует TACO class IDs → наши class IDs в .txt файлах разметки.
    TACO классы берём из classes.txt в корне датасета.
    """
    # Читаем оригинальные классы TACO
    with open(taco_class_file) as f:
        taco_classes = [l.strip() for l in f if l.strip()]

    # Строим маппинг: taco_id → наш id
    our_class_idx = {c: i for i, c in enumerate(OUR_CLASSES)}
    remap = {}
    for i, tc in enumerate(taco_classes):
        our_cat = TACO_CATEGORY_MAP.get(tc)
        if our_cat and our_cat in our_class_idx:
            remap[i] = our_class_idx[our_cat]

    print(f'[INFO] Маппинг классов: {len(remap)}/{len(taco_classes)} TACO → наши')

    # Переписываем .txt файлы
    remapped = 0
    skipped = 0
    for split in ('train', 'valid', 'test'):
        lbl_dir = os.path.join(data_dir, split, 'labels')
        if not os.path.isdir(lbl_dir):
            continue
        for fname in os.listdir(lbl_dir):
            if not fname.endswith('.txt'):
                continue
            path = os.path.join(lbl_dir, fname)
            new_lines = []
            with open(path) as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    old_id = int(parts[0])
                    if old_id in remap:
                        new_lines.append(f'{remap[old_id]} ' + ' '.join(parts[1:]))
                        remapped += 1
                    else:
                        skipped += 1
            with open(path, 'w') as f:
                f.write('\n'.join(new_lines) + ('\n' if new_lines else ''))

    print(f'[INFO] Переразмечено: {remapped} боксов, пропущено: {skipped}')


def train(data_yaml: str, output_path: str, epochs: int = EPOCHS, batch: int = BATCH):
    """Запускает fine-tuning YOLOv8n на TACO."""
    print(f'\n[INFO] Загрузка базовой модели: {BASE_MODEL}')
    model = YOLO(BASE_MODEL)

    print(f'[INFO] Запуск обучения: epochs={epochs}, imgsz={IMGSZ}, batch={batch}')
    print(f'[INFO] Устройство: {DEVICE}')
    print('[INFO] Это займёт 2-4 часа на GPU или ~12ч на CPU...\n')

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=IMGSZ,
        batch=batch,
        patience=PATIENCE,
        device=DEVICE,
        project=os.path.join(PACKAGE_DIR, 'data', 'runs'),
        name='taco_finetune',
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        augment=True,
        mixup=0.1,
        copy_paste=0.1,
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        verbose=True,
    )

    # Копируем лучший checkpoint в models/
    best = os.path.join(
        PACKAGE_DIR, 'data', 'runs', 'taco_finetune', 'weights', 'best.pt')
    if os.path.isfile(best):
        shutil.copy2(best, output_path)
        print(f'\n[OK] Модель сохранена: {output_path}')
        print(f'[INFO] Чтобы использовать её: скопируйте как models/yolov8n.pt')
        print(f'       cp {output_path} {BASE_MODEL}')
    else:
        print(f'[WARN] best.pt не найден, ищем last.pt...')
        last = best.replace('best.pt', 'last.pt')
        if os.path.isfile(last):
            shutil.copy2(last, output_path)
            print(f'[OK] Сохранён last.pt → {output_path}')

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune YOLOv8n на TACO dataset')
    parser.add_argument(
        '--api-key', default='',
        help='Roboflow API key (получить на https://app.roboflow.com → Settings → API)')
    parser.add_argument(
        '--data-dir', default=DATA_DIR,
        help=f'Папка с TACO датасетом (default: {DATA_DIR})')
    parser.add_argument(
        '--skip-download', action='store_true',
        help='Пропустить загрузку (если датасет уже скачан в --data-dir)')
    parser.add_argument(
        '--epochs', type=int, default=EPOCHS)
    parser.add_argument(
        '--batch', type=int, default=BATCH)
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Шаг 1: Загрузка датасета ─────────────────────────────────────────── #
    if not args.skip_download:
        if not args.api_key:
            print(
                '\n[INFO] Для загрузки TACO нужен Roboflow API key.\n'
                '  1. Зарегистрируйтесь на https://app.roboflow.com\n'
                '  2. Settings → API → скопируйте Private API Key\n'
                '  3. Запустите: python3 scripts/train_yolo.py --api-key YOUR_KEY\n'
                '\n'
                'Или скачайте вручную:\n'
                '  Откройте https://universe.roboflow.com/material-identification/'
                'taco-trash-annotations-in-context\n'
                '  Export → YOLOv8 → Download ZIP → распакуйте в:\n'
                f'  {DATA_DIR}\n'
                '\n'
                'Затем запустите:\n'
                f'  python3 scripts/train_yolo.py --skip-download\n'
            )
            sys.exit(0)
        download_taco_roboflow(args.api_key, args.data_dir)

    # ── Шаг 2: Проверка структуры ─────────────────────────────────────────── #
    expected = [
        os.path.join(args.data_dir, 'train', 'images'),
        os.path.join(args.data_dir, 'valid', 'images'),
    ]
    for d in expected:
        if not os.path.isdir(d):
            print(f'[ERROR] Папка не найдена: {d}')
            print(f'[INFO] Проверьте структуру датасета в {args.data_dir}')
            sys.exit(1)

    # ── Шаг 3: Фиксируем path в data.yaml (Roboflow пишет relative пути) ─── #
    import yaml
    yaml_path = os.path.join(args.data_dir, 'data.yaml')
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    cfg['path'] = args.data_dir
    cfg['train'] = 'train/images'
    cfg['val']   = 'valid/images'
    cfg['test']  = 'test/images'
    with open(yaml_path, 'w') as f:
        yaml.dump(cfg, f, allow_unicode=True)
    print(f'[INFO] data.yaml обновлён: {len(cfg["names"])} классов TACO')

    # ── Шаг 5: Обучение ──────────────────────────────────────────────────── #
    train(yaml_path, OUTPUT_MODEL, epochs=args.epochs, batch=args.batch)

    print('\n=== Готово ===')
    print(f'Обученная модель: {OUTPUT_MODEL}')
    print(f'Чтобы использовать: cp {OUTPUT_MODEL} {BASE_MODEL}')
    print('Затем перезапустите симуляцию.')


if __name__ == '__main__':
    main()
