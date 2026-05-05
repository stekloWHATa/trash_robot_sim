#!/usr/bin/env python3
"""
train_yolo.py — дообучение YOLOv8n на нескольких датасетах мусора.

Поддерживает слияние произвольного количества Roboflow-датасетов:
  - Скачивает каждый датасет
  - Приводит классы к целевой таксономии из config/trash_classes.yaml
  - Перемаппирует ID меток (.txt) под единую нумерацию
  - Объединяет train/valid/test сплиты
  - Обучает YOLOv8n на объединённом датасете

Использование:
  pip install ultralytics roboflow

  # Скачать и обучить (датасеты заданы в DATASETS ниже):
  python3 scripts/train_yolo.py --epochs 50

  # Если датасеты уже скачаны:
  python3 scripts/train_yolo.py --skip-download --prepare-only
"""

import os, sys, shutil, argparse, glob, json, re

# ── Пути ─────────────────────────────────────────────────────────────────── #
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR  = os.path.dirname(SCRIPT_DIR)
MODELS_DIR   = os.path.join(PACKAGE_DIR, 'models')
BASE_MODEL   = os.path.join(MODELS_DIR, 'yolov8n.pt')
OUTPUT_MODEL = os.path.join(MODELS_DIR, 'yolov8n_trash.pt')
DATA_ROOT    = os.path.join(PACKAGE_DIR, 'data')
MERGED_DIR   = os.path.join(DATA_ROOT, 'merged_v2')
CLASS_CONFIG = os.path.join(PACKAGE_DIR, 'config', 'trash_classes.yaml')

# ── Параметры обучения ────────────────────────────────────────────────────── #
EPOCHS   = 100
IMGSZ    = 640
BATCH    = 16
PATIENCE = 10
DEVICE   = '0'   # GPU. Поменяй на 'cpu' если нет NVIDIA

# ── Датасеты для слияния ──────────────────────────────────────────────────── #
# Формат: (workspace, project, version, local_folder_name)
DATASETS = [
    ('material-identification',    'garbage-classification-3',  2, 'garbage_class3'),
    ('garbage-segregation-bagn8',  'garbage-segregation-yyhof', 1, 'garbage_segregation'),
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


def _norm_name(name):
    """Нормализация имени класса для устойчивого mapping."""
    return re.sub(r'[^a-z0-9]+', '', str(name).lower())


def _names_to_list(names):
    if isinstance(names, dict):
        return [names[i] for i in sorted(names, key=lambda x: int(x))]
    return list(names or [])


def load_class_config(path=CLASS_CONFIG):
    """Читает целевую таксономию и строит alias -> canonical map."""
    if not path:
        return None
    if not os.path.isfile(path):
        print(f'[WARN] class config не найден: {path}; классы будут объединены как есть')
        return None

    cfg = read_yaml(path)
    classes = cfg.get('canonical_classes', [])
    if not classes:
        raise ValueError(f'canonical_classes пустой в {path}')

    class_set = set(classes)
    aliases = {}
    for cls in classes:
        aliases[_norm_name(cls)] = cls
    for cls, names in cfg.get('aliases', {}).items():
        if cls not in class_set:
            raise ValueError(f'aliases содержит неизвестный canonical class: {cls}')
        for name in names or []:
            aliases[_norm_name(name)] = cls

    ignored = {_norm_name(n) for n in cfg.get('ignored_classes', [])}
    return {
        'path': path,
        'classes': classes,
        'aliases': aliases,
        'ignored': ignored,
    }


def _map_class(name, class_config):
    if class_config is None:
        return str(name)
    key = _norm_name(name)
    if key in class_config['aliases']:
        return class_config['aliases'][key]
    return None


def _remap_yolo_annotation(parts, new_id):
    """Возвращает YOLO bbox строку; segmentation polygon сворачивает в bbox."""
    if len(parts) == 5:
        return f'{new_id} ' + ' '.join(parts[1:])
    if len(parts) > 5 and (len(parts) - 1) % 2 == 0:
        try:
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            return None
        xs = coords[0::2]
        ys = coords[1::2]
        x0, x1 = max(0.0, min(xs)), min(1.0, max(xs))
        y0, y1 = max(0.0, min(ys)), min(1.0, max(ys))
        w, h = x1 - x0, y1 - y0
        if w <= 0.0 or h <= 0.0:
            return None
        xc = x0 + w / 2.0
        yc = y0 + h / 2.0
        return f'{new_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}'
    return None


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


def merge_datasets(dataset_dirs, output_dir=None, class_config=None):
    """
    Объединяет несколько датасетов в один.
    Возвращает путь к объединённому data.yaml.
    """
    output_dir = output_dir or MERGED_DIR

    # ── 1. Собираем все уникальные классы ────────────────────────────────── #
    all_classes = list(class_config['classes']) if class_config else []
    seen = set()
    ds_configs = []
    stats = {
        'output_dir': output_dir,
        'source_datasets': [],
        'classes': all_classes,
        'images_by_split': {'train': 0, 'valid': 0, 'test': 0},
        'boxes_total': 0,
        'boxes_kept': 0,
        'boxes_skipped': 0,
        'invalid_labels': 0,
        'skipped_by_class': {},
        'boxes_by_class': {c: 0 for c in all_classes},
    }

    for d in dataset_dirs:
        yaml_path = os.path.join(d, 'data.yaml')
        if not os.path.isfile(yaml_path):
            print(f'[WARN] data.yaml не найден: {d}')
            continue
        cfg = fix_yaml_paths(d)
        names = _names_to_list(cfg.get('names', []))
        ds_configs.append((d, names))
        stats['source_datasets'].append({
            'path': d,
            'classes': names,
            'count': len(names),
        })
        if class_config is None:
            for n in names:
                nl = _norm_name(n)
                if nl not in seen:
                    seen.add(nl)
                    all_classes.append(n)
        print(f'[INFO] {os.path.basename(d)}: {len(names)} классов — {names}')

    if class_config is None:
        stats['classes'] = all_classes
        stats['boxes_by_class'] = {c: 0 for c in all_classes}

    print(f'\n[INFO] Объединённых классов: {len(all_classes)}')
    for i, c in enumerate(all_classes):
        print(f'  {i:2d}: {c}')

    # ── 2. Строим таблицу перемаппинга для каждого датасета ──────────────── #
    unified_idx = {_norm_name(c): i for i, c in enumerate(all_classes)}

    # ── 3. Копируем изображения и перемаппируем метки ────────────────────── #
    shutil.rmtree(output_dir, ignore_errors=True)
    for split in ('train', 'valid', 'test'):
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)

    total_imgs = 0
    for d, names in ds_configs:
        ds_name = os.path.basename(d)
        # Строим маппинг: старый ID → новый ID
        remap = {}
        for old_id, name in enumerate(names):
            mapped = _map_class(name, class_config)
            if mapped is None:
                continue
            new_id = unified_idx.get(_norm_name(mapped))
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
                            parts = line.strip().split()
                            if not parts:
                                continue
                            try:
                                old_id = int(parts[0])
                            except ValueError:
                                stats['invalid_labels'] += 1
                                continue
                            stats['boxes_total'] += 1
                            if old_id in remap:
                                new_id = remap[old_id]
                                cls_name = all_classes[new_id]
                                new_line = _remap_yolo_annotation(parts, new_id)
                                if new_line is None:
                                    stats['invalid_labels'] += 1
                                    continue
                                new_lines.append(new_line)
                                stats['boxes_kept'] += 1
                                stats['boxes_by_class'][cls_name] += 1
                            else:
                                old_name = names[old_id] if 0 <= old_id < len(names) else f'__invalid_{old_id}'
                                stats['boxes_skipped'] += 1
                                stats['skipped_by_class'][old_name] = (
                                    stats['skipped_by_class'].get(old_name, 0) + 1
                                )
                with open(dst_lbl, 'w') as f:
                    f.write('\n'.join(new_lines) + ('\n' if new_lines else ''))

                total_imgs += 1
                stats['images_by_split'][split] += 1

    print(f'\n[INFO] Скопировано изображений: {total_imgs}')
    print(f'[INFO] BBox: kept={stats["boxes_kept"]}, skipped={stats["boxes_skipped"]}')

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
    write_yaml(os.path.join(output_dir, 'merge_report.yaml'), stats)
    print(f'[INFO] Объединённый датасет: {merged_yaml}')
    return merged_yaml


def convert_coco_to_yolo(coco_json, images_dir, output_dir, split='train',
                         class_config=None, copy_images=True):
    """Конвертирует COCO/TACO annotations в YOLO-папку со split/images|labels."""
    with open(coco_json) as f:
        coco = json.load(f)

    categories = {int(c['id']): c['name'] for c in coco.get('categories', [])}
    if class_config:
        classes = list(class_config['classes'])
    else:
        classes = []
        seen = set()
        for _, name in sorted(categories.items()):
            key = _norm_name(name)
            if key not in seen:
                seen.add(key)
                classes.append(name)
    class_idx = {_norm_name(c): i for i, c in enumerate(classes)}

    out_img_dir = os.path.join(output_dir, split, 'images')
    out_lbl_dir = os.path.join(output_dir, split, 'labels')
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    images = {int(img['id']): img for img in coco.get('images', [])}
    labels_by_img = {img_id: [] for img_id in images}
    skipped = {}

    for ann in coco.get('annotations', []):
        if ann.get('iscrowd', 0):
            continue
        img = images.get(int(ann['image_id']))
        if not img:
            continue
        cat_name = categories.get(int(ann['category_id']), '')
        mapped = _map_class(cat_name, class_config)
        if mapped is None:
            skipped[cat_name] = skipped.get(cat_name, 0) + 1
            continue
        cls_id = class_idx[_norm_name(mapped)]
        x, y, w, h = [float(v) for v in ann['bbox']]
        iw, ih = float(img['width']), float(img['height'])
        x0 = max(0.0, min(iw, x))
        y0 = max(0.0, min(ih, y))
        x1 = max(0.0, min(iw, x + w))
        y1 = max(0.0, min(ih, y + h))
        bw, bh = x1 - x0, y1 - y0
        if bw <= 0.0 or bh <= 0.0:
            continue
        xc = (x0 + bw / 2.0) / iw
        yc = (y0 + bh / 2.0) / ih
        labels_by_img[int(ann['image_id'])].append(
            f'{cls_id} {xc:.6f} {yc:.6f} {bw / iw:.6f} {bh / ih:.6f}'
        )

    for img_id, img in images.items():
        src = os.path.join(images_dir, img['file_name'])
        stem = os.path.splitext(os.path.basename(img['file_name']))[0]
        ext = os.path.splitext(img['file_name'])[1] or '.jpg'
        if copy_images and os.path.isfile(src):
            shutil.copy2(src, os.path.join(out_img_dir, stem + ext))
        with open(os.path.join(out_lbl_dir, stem + '.txt'), 'w') as f:
            lines = labels_by_img.get(img_id, [])
            f.write('\n'.join(lines) + ('\n' if lines else ''))

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
        'source': coco_json,
        'images': len(images),
        'skipped_by_class': skipped,
        'classes': classes,
    })
    return data_yaml


def train(data_yaml, output_path, epochs=EPOCHS, batch=BATCH):
    print(f'\n[INFO] Модель: {BASE_MODEL}')
    print(f'[INFO] epochs={epochs}, batch={batch}, device={DEVICE}')

    try:
        from ultralytics import YOLO
    except ImportError:
        print('[ERROR] pip install ultralytics')
        sys.exit(1)

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
    parser.add_argument('--prepare-only',  action='store_true')
    parser.add_argument('--class-config',  default=CLASS_CONFIG)
    parser.add_argument('--output-dir',    default=MERGED_DIR)
    parser.add_argument('--extra-dataset', action='append', default=[])
    parser.add_argument('--coco-json',     default='')
    parser.add_argument('--coco-images',   default='')
    parser.add_argument('--coco-output',   default='')
    parser.add_argument('--coco-split',    default='train')
    parser.add_argument('--epochs',        type=int, default=EPOCHS)
    parser.add_argument('--batch',         type=int, default=BATCH)
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_ROOT,  exist_ok=True)
    class_config = load_class_config(args.class_config)

    if args.coco_json:
        if not args.coco_images or not args.coco_output:
            print('[ERROR] Для COCO/TACO нужны --coco-images и --coco-output')
            sys.exit(1)
        convert_coco_to_yolo(
            args.coco_json,
            args.coco_images,
            args.coco_output,
            split=args.coco_split,
            class_config=class_config,
            copy_images=True,
        )
        if args.prepare_only:
            return

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

    for extra in args.extra_dataset:
        if os.path.isdir(extra):
            dataset_dirs.append(extra)
            print(f'[INFO] Доп. датасет: {extra}')
        else:
            print(f'[WARN] Доп. датасет не найден: {extra}')

    if not dataset_dirs:
        print('[ERROR] Нет датасетов для обучения')
        sys.exit(1)

    # ── Слияние ───────────────────────────────────────────────────────────── #
    print(f'\n[INFO] Объединяем {len(dataset_dirs)} датасет(ов)...')
    merged_yaml = merge_datasets(
        dataset_dirs,
        output_dir=args.output_dir,
        class_config=class_config,
    )

    if args.prepare_only:
        print('[OK] Датасет подготовлен, обучение пропущено (--prepare-only)')
        return

    # ── Обучение ──────────────────────────────────────────────────────────── #
    train(merged_yaml, OUTPUT_MODEL, epochs=args.epochs, batch=args.batch)


if __name__ == '__main__':
    main()
