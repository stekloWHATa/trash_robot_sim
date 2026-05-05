#!/usr/bin/env python3
"""Быстрый аудит YOLO-датасета перед обучением."""

import argparse
import json
import math
import os
from pathlib import Path

import yaml


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def _read_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _names_to_list(names):
    if isinstance(names, dict):
        return [names[i] for i in sorted(names, key=lambda x: int(x))]
    return list(names or [])


def _split_path(root, value):
    path = Path(value)
    if not path.is_absolute():
        path = Path(root) / path
    return path


def _median(values):
    if not values:
        return None
    vals = sorted(values)
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def audit_yolo_dataset(data_yaml):
    cfg = _read_yaml(data_yaml)
    root = cfg.get('path') or os.path.dirname(os.path.abspath(data_yaml))
    names = _names_to_list(cfg.get('names', []))
    nc = int(cfg.get('nc', len(names)))

    audit = {
        'data_yaml': os.path.abspath(data_yaml),
        'path': os.path.abspath(root),
        'nc': nc,
        'names': names,
        'totals': {
            'images': 0,
            'labels': 0,
            'boxes': 0,
            'empty_labels': 0,
            'orphan_images': 0,
            'orphan_labels': 0,
            'invalid_labels': 0,
        },
        'class_counts': {name: 0 for name in names},
        'splits': {},
    }

    for split_key, split_name in [('train', 'train'), ('val', 'valid'), ('test', 'test')]:
        img_dir = _split_path(root, cfg.get(split_key, f'{split_name}/images'))
        lbl_dir = Path(str(img_dir).replace('/images', '/labels'))

        images = sorted(p for p in img_dir.glob('*') if p.suffix.lower() in IMAGE_EXTS) if img_dir.is_dir() else []
        labels = sorted(lbl_dir.glob('*.txt')) if lbl_dir.is_dir() else []
        image_stems = {p.stem for p in images}
        label_stems = {p.stem for p in labels}

        split = {
            'images': len(images),
            'labels': len(labels),
            'boxes': 0,
            'empty_labels': 0,
            'orphan_images': len(image_stems - label_stems),
            'orphan_labels': len(label_stems - image_stems),
            'invalid_labels': 0,
            'bbox_area': {
                'min': None,
                'max': None,
                'mean': None,
                'median': None,
            },
        }
        areas = []

        for label_path in labels:
            raw_lines = label_path.read_text().splitlines()
            lines = [line.strip() for line in raw_lines if line.strip()]
            if not lines:
                split['empty_labels'] += 1
            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    split['invalid_labels'] += 1
                    continue
                try:
                    cls = int(parts[0])
                    x, y, w, h = [float(v) for v in parts[1:]]
                except ValueError:
                    split['invalid_labels'] += 1
                    continue
                if cls < 0 or cls >= nc or not all(math.isfinite(v) for v in (x, y, w, h)):
                    split['invalid_labels'] += 1
                    continue
                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    split['invalid_labels'] += 1
                    continue
                split['boxes'] += 1
                areas.append(w * h)
                if cls < len(names):
                    audit['class_counts'][names[cls]] += 1

        if areas:
            split['bbox_area'] = {
                'min': min(areas),
                'max': max(areas),
                'mean': sum(areas) / len(areas),
                'median': _median(areas),
            }

        audit['splits'][split_key] = split
        for key in audit['totals']:
            audit['totals'][key] += split.get(key, 0)

    return audit


def write_reports(audit, json_path=None, md_path=None):
    if json_path:
        with open(json_path, 'w') as f:
            json.dump(audit, f, ensure_ascii=False, indent=2)

    if md_path:
        lines = [
            '# YOLO Dataset Audit',
            '',
            f'- data: `{audit["data_yaml"]}`',
            f'- path: `{audit["path"]}`',
            f'- classes: {audit["nc"]}',
            '',
            '## Totals',
            '',
        ]
        for key, value in audit['totals'].items():
            lines.append(f'- {key}: {value}')
        lines.extend(['', '## Splits', ''])
        for split_name, split in audit['splits'].items():
            lines.append(f'### {split_name}')
            for key, value in split.items():
                if key == 'bbox_area':
                    continue
                lines.append(f'- {key}: {value}')
            lines.append('')
        lines.extend(['## Class Counts', ''])
        for name, count in sorted(audit['class_counts'].items(), key=lambda x: -x[1]):
            lines.append(f'- `{name}`: {count}')
        with open(md_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Audit YOLO dataset data.yaml')
    parser.add_argument('data_yaml')
    parser.add_argument('--json', default='')
    parser.add_argument('--md', default='')
    args = parser.parse_args()

    audit = audit_yolo_dataset(args.data_yaml)
    root = audit['path']
    json_path = args.json or os.path.join(root, 'audit.json')
    md_path = args.md or os.path.join(root, 'audit.md')
    write_reports(audit, json_path=json_path, md_path=md_path)
    print(f'[OK] audit json: {json_path}')
    print(f'[OK] audit md:   {md_path}')
    if audit['totals']['invalid_labels']:
        sys_exit = 1
    else:
        sys_exit = 0
    raise SystemExit(sys_exit)


if __name__ == '__main__':
    main()
