#!/usr/bin/env python3
"""
train_yolo.py — дообучение YOLOv8n на нескольких датасетах мусора.

Поддерживает слияние произвольного количества Roboflow-датасетов:
  - Скачивает каждый датасет
  - Объединяет классы в единый список (дедупликация по имени)
  - Перемаппирует ID меток (.txt) под единую нумерацию
  - Объединяет train/valid/test сплиты
  - Обучает YOLOv8n на объединённом датасете

Использование:
  pip install ultralytics roboflow

  # Скачать и обучить (датасеты заданы в DATASETS ниже):
  python3 scripts/train_yolo.py --epochs 50

  # Если датасеты уже скачаны:
  python3 scripts/train_yolo.py --skip-download --epochs 50
"""

import os, sys, shutil, argparse, glob

try:
    from ultralytics import YOLO
except ImportError:
    print('[ERROR] pip install ultralytics'); sys.exit(1)

# ── Пути ─────────────────────────────────────────────────────────────────── #
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR  = os.path.dirname(SCRIPT_DIR)
MODELS_DIR   = os.path.join(PACKAGE_DIR, 'models')
BASE_MODEL   = os.path.join(MODELS_DIR, 'yolov8n.pt')
OUTPUT_MODEL = os.path.join(MODELS_DIR, 'yolov8n_trash.pt')
DATA_ROOT    = os.path.join(PACKAGE_DIR, 'data')
MERGED_DIR   = os.path.join(DATA_ROOT, 'merged')

# ── Параметры обучения ────────────────────────────────────────────────────── #
EPOCHS   = 100
IMGSZ    = 640
BATCH    = 16
PATIENCE = 10
DEVICE   = '0'   # GPU. Поменяй на 'cpu' если нет NVIDIA

# ── Датасеты для слияния ──────────────────────────────────────────────────── #
# Формат: (workspace, project, version, local_folder_name)
DATASETS = [
    ('material-identification',    'garbage-classification-3',  1, 'garbage_class3'),
    ('garbage-segregation-bagn8',  'garbage-segregation-yyhof', 2, 'garbage_segregation'),
]


# ─────────────────────────────────────────────────────────────────────────── #

def download_dataset(api_key, workspace, project, version, dest):
    """Скачивает один датасет с Roboflow."""
    import time, zipfile
    try:
        from roboflow import Roboflow
    except ImportError:
        print('[ERROR] pip install roboflow'); sys.exit(1)

    os.makedirs(dest, exist_ok=True)
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    ver  = proj.version(version)

    print(f'[INFO] Генерация экспорта {workspace}/{project} v{version}...')
    ver.export('yolov8')

    zip_path = os.path.join(dest, 'roboflow.zip')
    for attempt in range(1, 10):
        if os.path.isfile(zip_path):
            os.remove(zip_path)
        print(f'  Скачивание, попытка {attempt}/9...')
        try:
            ver.download('yolov8', location=dest, overwrite=True)
            if os.path.isfile(zip_path):
                with zipfile.ZipFile(zip_path):
                    pass
            print(f'  [OK] {dest}')
            return
        except Exception as e:
            if 'BadZipFile' in type(e).__name__ or 'not a zip' in str(e).lower():
                print(f'  Экспорт ещё генерируется, ждём 20с...')
                time.sleep(20)
            else:
                print(f'  [ERROR] {e}'); sys.exit(1)

    print(f'[ERROR] Не удалось скачать {workspace}/{project}')
    sys.exit(1)


def read_yaml(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def write_yaml(path, data):
    import yaml
    with open(path, 'w') as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)


def fix_yaml_paths(data_dir):
    """Фиксирует относительные пути в data.yaml от Roboflow."""
    yaml_path = os.path.join(data_dir, 'data.yaml')
    if not os.path.isfile(yaml_path):
        return None
    cfg = read_yaml(yaml_path)
    cfg['path']  = data_dir
    cfg['train'] = 'train/images'
    cfg['val']   = 'valid/images'
    cfg['test']  = 'test/images'
    write_yaml(yaml_path, cfg)
    return cfg


def merge_datasets(dataset_dirs):
    """
    Объединяет несколько датасетов в один.
    Возвращает путь к объединённому data.yaml.
    """
    # ── 1. Собираем все уникальные классы ────────────────────────────────── #
    all_classes = []   # сохраняем порядок первого появления
    seen = set()
    ds_configs = []

    for d in dataset_dirs:
        yaml_path = os.path.join(d, 'data.yaml')
        if not os.path.isfile(yaml_path):
            print(f'[WARN] data.yaml не найден: {d}')
            continue
        cfg = fix_yaml_paths(d)
        names = cfg.get('names', [])
        if isinstance(names, dict):
            names = [names[i] for i in sorted(names)]
        ds_configs.append((d, names))
        for n in names:
            nl = n.lower().strip()
            if nl not in seen:
                seen.add(nl)
                all_classes.append(n)
        print(f'[INFO] {os.path.basename(d)}: {len(names)} классов — {names}')

    print(f'\n[INFO] Объединённых классов: {len(all_classes)}')
    for i, c in enumerate(all_classes):
        print(f'  {i:2d}: {c}')

    # ── 2. Строим таблицу перемаппинга для каждого датасета ──────────────── #
    unified_idx = {c.lower().strip(): i for i, c in enumerate(all_classes)}

    # ── 3. Копируем изображения и перемаппируем метки ────────────────────── #
    shutil.rmtree(MERGED_DIR, ignore_errors=True)
    for split in ('train', 'valid', 'test'):
        os.makedirs(os.path.join(MERGED_DIR, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(MERGED_DIR, split, 'labels'), exist_ok=True)

    total_imgs = 0
    for d, names in ds_configs:
        ds_name = os.path.basename(d)
        # Строим маппинг: старый ID → новый ID
        remap = {}
        for old_id, name in enumerate(names):
            new_id = unified_idx.get(name.lower().strip())
            if new_id is not None:
                remap[old_id] = new_id

        for split in ('train', 'valid', 'test'):
            img_dir = os.path.join(d, split, 'images')
            lbl_dir = os.path.join(d, split, 'labels')
            if not os.path.isdir(img_dir):
                continue

            imgs = glob.glob(os.path.join(img_dir, '*'))
            for img_path in imgs:
                fname = os.path.basename(img_path)
                stem  = os.path.splitext(fname)[0]
                # Уникальное имя: датасет + оригинальное имя
                new_stem = f'{ds_name}__{stem}'
                ext = os.path.splitext(fname)[1]

                # Копируем изображение
                dst_img = os.path.join(MERGED_DIR, split, 'images', new_stem + ext)
                shutil.copy2(img_path, dst_img)

                # Перемаппируем метку
                src_lbl = os.path.join(lbl_dir, stem + '.txt')
                dst_lbl = os.path.join(MERGED_DIR, split, 'labels', new_stem + '.txt')
                new_lines = []
                if os.path.isfile(src_lbl):
                    with open(src_lbl) as f:
                        for line in f:
                            parts = line.strip().split()
                            if not parts:
                                continue
                            old_id = int(parts[0])
                            if old_id in remap:
                                new_lines.append(f'{remap[old_id]} ' + ' '.join(parts[1:]))
                with open(dst_lbl, 'w') as f:
                    f.write('\n'.join(new_lines) + ('\n' if new_lines else ''))

                total_imgs += 1

    print(f'\n[INFO] Скопировано изображений: {total_imgs}')

    # ── 4. Создаём data.yaml для объединённого датасета ───────────────────── #
    merged_yaml = os.path.join(MERGED_DIR, 'data.yaml')
    write_yaml(merged_yaml, {
        'path':  MERGED_DIR,
        'train': 'train/images',
        'val':   'valid/images',
        'test':  'test/images',
        'nc':    len(all_classes),
        'names': all_classes,
    })
    print(f'[INFO] Объединённый датасет: {merged_yaml}')
    return merged_yaml


def train(data_yaml, output_path, epochs=EPOCHS, batch=BATCH):
    print(f'\n[INFO] Модель: {BASE_MODEL}')
    print(f'[INFO] epochs={epochs}, batch={batch}, device={DEVICE}')
    model = YOLO(BASE_MODEL)
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=IMGSZ,
        batch=batch,
        patience=PATIENCE,
        device=DEVICE,
        project=os.path.join(DATA_ROOT, 'runs'),
        name='trash_finetune',
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
        fliplr=0.5,
        mosaic=1.0,
        verbose=True,
    )
    best = os.path.join(DATA_ROOT, 'runs', 'trash_finetune', 'weights', 'best.pt')
    src  = best if os.path.isfile(best) else best.replace('best.pt', 'last.pt')
    if os.path.isfile(src):
        shutil.copy2(src, output_path)
        print(f'\n[OK] Модель: {output_path}')
        print(f'     Скопируй как yolov8n.pt:')
        print(f'     cp {output_path} {BASE_MODEL}')
    else:
        print('[WARN] Весовой файл не найден')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-key',       default='')
    parser.add_argument('--skip-download', action='store_true')
    parser.add_argument('--epochs',        type=int, default=EPOCHS)
    parser.add_argument('--batch',         type=int, default=BATCH)
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_ROOT,  exist_ok=True)

    dataset_dirs = []

    # ── Скачивание ────────────────────────────────────────────────────────── #
    if not args.skip_download:
        if not args.api_key:
            print('\n[INFO] Нужен Roboflow API key (Settings → API → Private key)')
            print('  python3 train_yolo.py --api-key YOUR_KEY\n')
            print('Или скачай вручную и запусти --skip-download')
            sys.exit(0)
        for workspace, project, version, folder in DATASETS:
            dest = os.path.join(DATA_ROOT, folder)
            print(f'\n[INFO] Датасет: {workspace}/{project} v{version} → {folder}')
            download_dataset(args.api_key, workspace, project, version, dest)
            dataset_dirs.append(dest)
    else:
        for _, _, _, folder in DATASETS:
            dest = os.path.join(DATA_ROOT, folder)
            if os.path.isdir(dest):
                dataset_dirs.append(dest)
                print(f'[INFO] Найден: {dest}')
            else:
                print(f'[WARN] Не найден: {dest}')

    if not dataset_dirs:
        print('[ERROR] Нет датасетов для обучения')
        sys.exit(1)

    # ── Слияние ───────────────────────────────────────────────────────────── #
    print(f'\n[INFO] Объединяем {len(dataset_dirs)} датасет(ов)...')
    merged_yaml = merge_datasets(dataset_dirs)

    # ── Обучение ──────────────────────────────────────────────────────────── #
    train(merged_yaml, OUTPUT_MODEL, epochs=args.epochs, batch=args.batch)


if __name__ == '__main__':
    main()
