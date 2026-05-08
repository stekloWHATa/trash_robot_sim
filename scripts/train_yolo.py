#!/usr/bin/env python3
"""
train_yolo.py — дообучение YOLOv8s на датасетах бытового мусора.

Поддерживает слияние произвольного количества Roboflow-датасетов:
  - Скачивает каждый датасет
  - Объединяет классы в единый список (дедупликация по имени)
  - Перемаппирует ID меток (.txt) под единую нумерацию
  - Объединяет train/valid/test сплиты
  - Обучает YOLOv8s на объединённом датасете

Использование:
  pip install ultralytics roboflow

  # Скачать и обучить (датасеты заданы в DATASETS ниже):
  python3 scripts/train_yolo.py --epochs 50

  # Если датасеты уже скачаны:
  python3 scripts/train_yolo.py --skip-download --epochs 50

  # Только собрать merged-набор из локальных датасетов:
  python3 scripts/train_yolo.py --skip-download --extra-dataset data/taco_yolo --prepare-only
"""

import os, sys, shutil, argparse, glob, json

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# ── Пути ─────────────────────────────────────────────────────────────────── #
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR  = os.path.dirname(SCRIPT_DIR)
MODELS_DIR   = os.path.join(PACKAGE_DIR, 'models')
BASE_MODEL   = os.path.join(MODELS_DIR, 'yolov8s.pt')
OUTPUT_MODEL = os.path.join(MODELS_DIR, 'yolov8s_trash.pt')
DATA_ROOT    = os.path.join(PACKAGE_DIR, 'data')
MERGED_DIR   = os.path.join(DATA_ROOT, 'merged_v2')
DEFAULT_CLASS_CONFIG = os.path.join(PACKAGE_DIR, 'config', 'trash_classes_mvp.yaml')

# ── Параметры обучения ────────────────────────────────────────────────────── #
EPOCHS   = 120
IMGSZ    = 960
BATCH    = 16
PATIENCE = 25
DEVICE   = '0'   # GPU. Поменяй на 'cpu' если нет NVIDIA
RUN_NAME = 'trash_yolov8s_img960'

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


def _normalize_class_name(name):
    return str(name).strip().lower().replace('-', ' ').replace('_', ' ')


def load_class_config(path):
    cfg = read_yaml(path)
    classes = list(cfg.get('canonical_classes', []))
    ignored = {
        _normalize_class_name(name)
        for name in cfg.get('ignored_classes', [])
    }
    alias_to_class = {}
    for cls in classes:
        alias_to_class[_normalize_class_name(cls)] = cls
    for cls, aliases in (cfg.get('aliases', {}) or {}).items():
        canonical = cls if cls in classes else _normalize_class_name(cls).replace(' ', '_')
        if canonical in classes:
            alias_to_class[_normalize_class_name(cls)] = canonical
            for alias in aliases or []:
                alias_to_class[_normalize_class_name(alias)] = canonical
    return {
        'classes': classes,
        'alias_to_class': alias_to_class,
        'ignored': ignored,
    }


def _map_class(name, class_config):
    normalized = _normalize_class_name(name)
    if normalized in class_config.get('ignored', set()):
        return None
    return class_config.get('alias_to_class', {}).get(normalized)


def _yolo_polygon_to_bbox(values):
    coords = [float(v) for v in values]
    xs = coords[0::2]
    ys = coords[1::2]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    xc = (x1 + x2) / 2.0
    yc = (y1 + y2) / 2.0
    bw = x2 - x1
    bh = y2 - y1
    return [xc, yc, bw, bh]


def _convert_yolo_line(line, remap):
    parts = line.strip().split()
    if not parts:
        return None, None
    old_id = int(parts[0])
    if old_id not in remap:
        return None, old_id
    values = parts[1:]
    if len(values) == 4:
        return f'{remap[old_id]} ' + ' '.join(values), old_id
    elif len(values) >= 6 and len(values) % 2 == 0:
        bbox = _yolo_polygon_to_bbox(values)
    else:
        return None, old_id
    return (
        f'{remap[old_id]} '
        f'{bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}',
        old_id,
    )


def merge_datasets(dataset_dirs, class_config=None, output_dir=None):
    """
    Объединяет несколько датасетов в один.
    Возвращает путь к объединённому data.yaml.
    """
    output_dir = output_dir or MERGED_DIR

    # ── 1. Собираем все уникальные классы ────────────────────────────────── #
    all_classes = list(class_config['classes']) if class_config else []
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
        if not class_config:
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
    shutil.rmtree(output_dir, ignore_errors=True)
    for split in ('train', 'valid', 'test'):
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)

    total_imgs = 0
    boxes_kept = 0
    boxes_skipped = 0
    skipped_by_class = {}
    for d, names in ds_configs:
        ds_name = os.path.basename(d)
        # Строим маппинг: старый ID → новый ID
        remap = {}
        for old_id, name in enumerate(names):
            if class_config:
                mapped = _map_class(name, class_config)
                new_id = unified_idx.get(mapped.lower().strip()) if mapped else None
            else:
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
                dst_img = os.path.join(output_dir, split, 'images', new_stem + ext)
                shutil.copy2(img_path, dst_img)

                # Перемаппируем метку
                src_lbl = os.path.join(lbl_dir, stem + '.txt')
                dst_lbl = os.path.join(output_dir, split, 'labels', new_stem + '.txt')
                new_lines = []
                if os.path.isfile(src_lbl):
                    with open(src_lbl) as f:
                        for line in f:
                            converted, old_id = _convert_yolo_line(line, remap)
                            if converted is not None:
                                new_lines.append(converted)
                                boxes_kept += 1
                            elif old_id is not None:
                                boxes_skipped += 1
                                if 0 <= old_id < len(names):
                                    skipped_name = names[old_id]
                                else:
                                    skipped_name = str(old_id)
                                skipped_by_class[skipped_name] = skipped_by_class.get(skipped_name, 0) + 1
                with open(dst_lbl, 'w') as f:
                    f.write('\n'.join(new_lines) + ('\n' if new_lines else ''))

                total_imgs += 1

    print(f'\n[INFO] Скопировано изображений: {total_imgs}')

    # ── 4. Создаём data.yaml для объединённого датасета ───────────────────── #
    merged_yaml = os.path.join(output_dir, 'data.yaml')
    write_yaml(merged_yaml, {
        'path':  output_dir,
        'train': 'train/images',
        'val':   'valid/images',
        'test':  'test/images',
        'nc':    len(all_classes),
        'names': all_classes,
    })
    write_yaml(os.path.join(output_dir, 'merge_report.yaml'), {
        'images': total_imgs,
        'boxes_kept': boxes_kept,
        'boxes_skipped': boxes_skipped,
        'skipped_by_class': skipped_by_class,
    })
    print(f'[INFO] Объединённый датасет: {merged_yaml}')
    return merged_yaml


def convert_coco_to_yolo(coco_json, images_dir, output_dir, split='train', class_config=None):
    with open(coco_json, encoding='utf-8') as f:
        coco = json.load(f)
    split = str(split or 'train').lower()
    classes = list(class_config['classes']) if class_config else [
        c['name'] for c in coco.get('categories', [])
    ]
    class_to_idx = {name: i for i, name in enumerate(classes)}
    cat_to_class = {}
    for cat in coco.get('categories', []):
        name = cat['name']
        mapped = _map_class(name, class_config) if class_config else name
        if mapped in class_to_idx:
            cat_to_class[cat['id']] = mapped

    image_by_id = {img['id']: img for img in coco.get('images', [])}
    labels_by_image = {img_id: [] for img_id in image_by_id}
    for ann in coco.get('annotations', []):
        mapped = cat_to_class.get(ann.get('category_id'))
        if mapped is None:
            continue
        img = image_by_id.get(ann.get('image_id'))
        if img is None:
            continue
        x, y, w, h = [float(v) for v in ann['bbox']]
        if img.get('width', 0) <= 0 or img.get('height', 0) <= 0:
            continue
        xc = (x + w / 2.0) / float(img['width'])
        yc = (y + h / 2.0) / float(img['height'])
        bw = w / float(img['width'])
        bh = h / float(img['height'])
        labels_by_image[img['id']].append(
            f'{class_to_idx[mapped]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}'
        )

    shutil.rmtree(output_dir, ignore_errors=True)
    for split_name in ('train', 'valid', 'test'):
        os.makedirs(os.path.join(output_dir, split_name, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split_name, 'labels'), exist_ok=True)

    sorted_images = sorted(image_by_id.items(), key=lambda item: str(item[0]))
    split_counts = {'train': 0, 'valid': 0, 'test': 0}
    box_counts = {'train': 0, 'valid': 0, 'test': 0}

    for idx, (img_id, img) in enumerate(sorted_images):
        if split == 'auto':
            frac = idx / max(1, len(sorted_images))
            out_split = 'train' if frac < 0.80 else ('valid' if frac < 0.90 else 'test')
        elif split in ('val', 'valid'):
            out_split = 'valid'
        elif split == 'test':
            out_split = 'test'
        else:
            out_split = 'train'

        img_out = os.path.join(output_dir, out_split, 'images')
        lbl_out = os.path.join(output_dir, out_split, 'labels')
        src = os.path.join(images_dir, img['file_name'])
        safe_name = img['file_name'].replace('\\', '/').replace('/', '__')
        dst = os.path.join(img_out, safe_name)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
        stem = os.path.splitext(safe_name)[0]
        lines = labels_by_image.get(img_id, [])
        with open(os.path.join(lbl_out, stem + '.txt'), 'w') as f:
            f.write('\n'.join(lines) + ('\n' if lines else ''))
        split_counts[out_split] += 1
        box_counts[out_split] += len(lines)

    data_yaml = os.path.join(output_dir, 'data.yaml')
    write_yaml(data_yaml, {
        'path': output_dir,
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': len(classes),
        'names': classes,
    })
    write_yaml(os.path.join(output_dir, f'{split}_coco_convert_report.yaml'), {
        'source_json': coco_json,
        'source_images': images_dir,
        'split_mode': split,
        'images': split_counts,
        'boxes': box_counts,
        'classes': classes,
    })
    return data_yaml


def _resolve_base_model(base_model: str) -> str:
    if os.path.isfile(base_model):
        return base_model
    basename = os.path.basename(base_model)
    if basename in {'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt'}:
        print(f'[WARN] Локальная base model не найдена: {base_model}; '
              f'передаю Ultralytics имя модели {basename!r}')
        return basename
    return base_model


def train(data_yaml, output_path, epochs=EPOCHS, batch=BATCH,
          base_model=BASE_MODEL, imgsz=IMGSZ, device=DEVICE,
          run_name=RUN_NAME):
    if YOLO is None:
        print('[ERROR] pip install ultralytics')
        sys.exit(1)
    base_model = _resolve_base_model(base_model)
    print(f'\n[INFO] Модель: {base_model}')
    print(f'[INFO] epochs={epochs}, batch={batch}, imgsz={imgsz}, device={device}')
    model = YOLO(base_model)
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=PATIENCE,
        device=device,
        project=os.path.join(DATA_ROOT, 'runs'),
        name=run_name,
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
    best = os.path.join(DATA_ROOT, 'runs', run_name, 'weights', 'best.pt')
    src  = best if os.path.isfile(best) else best.replace('best.pt', 'last.pt')
    if os.path.isfile(src):
        shutil.copy2(src, output_path)
        print(f'\n[OK] Модель: {output_path}')
        print(f'     Для detector.py укажи model_path или оставь default yolov8s_trash.pt.')
    else:
        print('[WARN] Весовой файл не найден')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-key',       default='')
    parser.add_argument('--skip-download', action='store_true')
    parser.add_argument('--epochs',        type=int, default=EPOCHS)
    parser.add_argument('--batch',         type=int, default=BATCH)
    parser.add_argument('--imgsz',         type=int, default=IMGSZ)
    parser.add_argument('--device',        default=DEVICE)
    parser.add_argument('--base-model',    default=BASE_MODEL)
    parser.add_argument('--output-model',  default=OUTPUT_MODEL)
    parser.add_argument('--run-name',      default=RUN_NAME)
    parser.add_argument('--prepare-only',  action='store_true')
    parser.add_argument('--extra-dataset', action='append', default=[],
                        help='Локальный YOLOv8 dataset dir с data.yaml. Можно повторять.')
    parser.add_argument('--only-extra-datasets', action='store_true',
                        help='Не добавлять стандартные DATASETS, использовать только --extra-dataset/--coco-json.')
    parser.add_argument('--output-dir',    default=MERGED_DIR,
                        help='Куда писать объединенный YOLO dataset.')
    parser.add_argument('--class-config', default=DEFAULT_CLASS_CONFIG)
    parser.add_argument('--coco-json',     default='')
    parser.add_argument('--coco-images',   default='')
    parser.add_argument('--coco-output',   default=os.path.join(DATA_ROOT, 'taco_yolo'))
    parser.add_argument('--coco-split',    default='auto')
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_ROOT,  exist_ok=True)

    class_config = None
    if args.class_config and os.path.isfile(args.class_config):
        class_config = load_class_config(args.class_config)
        print(f'[INFO] Class mapping: {args.class_config}')
    elif args.class_config:
        print(f'[WARN] Class config не найден: {args.class_config}')

    dataset_dirs = []

    # ── Отдельная COCO/TACO-конвертация ──────────────────────────────────── #
    if args.coco_json:
        if not args.coco_images:
            print('[ERROR] Для --coco-json нужен --coco-images')
            sys.exit(1)
        print(f'[INFO] COCO/TACO -> YOLO: {args.coco_json} → {args.coco_output}')
        convert_coco_to_yolo(
            args.coco_json,
            args.coco_images,
            args.coco_output,
            split=args.coco_split,
            class_config=class_config,
        )
        dataset_dirs.append(args.coco_output)
        if args.prepare_only and args.skip_download and not args.extra_dataset:
            print(f'[OK] Конвертация готова: {args.coco_output}')
            return

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
        if args.only_extra_datasets:
            print('[INFO] Стандартные DATASETS пропущены: --only-extra-datasets')
        else:
            for _, _, _, folder in DATASETS:
                dest = os.path.join(DATA_ROOT, folder)
                if os.path.isdir(dest):
                    dataset_dirs.append(dest)
                    print(f'[INFO] Найден: {dest}')
                else:
                    print(f'[WARN] Не найден: {dest}')

    for extra in args.extra_dataset:
        if os.path.isdir(extra):
            dataset_dirs.append(extra)
            print(f'[INFO] Extra dataset: {extra}')
        else:
            print(f'[WARN] Extra dataset не найден: {extra}')

    if not dataset_dirs:
        print('[ERROR] Нет датасетов для обучения')
        sys.exit(1)

    # ── Слияние ───────────────────────────────────────────────────────────── #
    print(f'\n[INFO] Объединяем {len(dataset_dirs)} датасет(ов)...')
    merged_yaml = merge_datasets(
        dataset_dirs,
        class_config=class_config,
        output_dir=args.output_dir,
    )

    if args.prepare_only:
        print('[OK] prepare-only: обучение не запускалось')
        return

    # ── Обучение ──────────────────────────────────────────────────────────── #
    train(
        merged_yaml,
        args.output_model,
        epochs=args.epochs,
        batch=args.batch,
        base_model=args.base_model,
        imgsz=args.imgsz,
        device=args.device,
        run_name=args.run_name,
    )


if __name__ == '__main__':
    main()
