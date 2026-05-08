#!/usr/bin/env python3
"""Check whether a YOLO dataset is ready for a high-quality YOLOv8s run."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit_dataset


DEFAULT_MIN_BOXES = {
    'cigarette_butt': 3000,
    'plastic_bottle': 1000,
    'aluminum_can': 1000,
    'plastic_bag': 1000,
    'cardboard_box': 1000,
    'paper_packaging': 500,
}


def _parse_min_boxes(items: list[str]) -> dict[str, int]:
    thresholds = dict(DEFAULT_MIN_BOXES)
    for item in items:
        if ':' not in item:
            raise ValueError(f'Bad --min-boxes value {item!r}, expected class:count')
        name, value = item.split(':', 1)
        thresholds[name.strip()] = int(value)
    return thresholds


def check_readiness(data_yaml: str | os.PathLike,
                    min_boxes: dict[str, int] | None = None) -> dict:
    audit = audit_dataset.audit_yolo_dataset(str(data_yaml))
    thresholds = min_boxes or DEFAULT_MIN_BOXES
    class_counts = audit['class_counts']

    problems = []
    warnings = []
    if audit['totals']['invalid_labels'] > 0:
        problems.append(
            f'invalid labels: {audit["totals"]["invalid_labels"]}'
        )
    if audit['totals']['orphan_labels'] > 0:
        problems.append(
            f'orphan labels: {audit["totals"]["orphan_labels"]}'
        )

    missing_classes = [name for name in thresholds if name not in class_counts]
    for name in missing_classes:
        problems.append(f'class missing from dataset: {name}')

    for name, minimum in thresholds.items():
        if name not in class_counts:
            continue
        count = int(class_counts[name])
        if count < minimum:
            problems.append(f'{name}: {count} boxes < required {minimum}')

    train_images = audit['splits'].get('train', {}).get('images', 0)
    valid_images = audit['splits'].get('val', {}).get('images', 0)
    test_images = audit['splits'].get('test', {}).get('images', 0)
    if train_images <= 0 or valid_images <= 0:
        problems.append('train/valid splits must both contain images')
    if test_images <= 0:
        warnings.append('test split is empty; keep a held-out sim or dataset benchmark')

    totals = audit['totals']
    ready = not problems
    return {
        'ready_for_yolov8s': ready,
        'data_yaml': audit['data_yaml'],
        'totals': totals,
        'class_counts': class_counts,
        'thresholds': thresholds,
        'problems': problems,
        'warnings': warnings,
        'recommendation': (
            'Ready for yolov8s/imgsz=960 training.'
            if ready
            else 'Fix dataset balance/labels before long GPU training.'
        ),
    }


def write_markdown(report: dict, path: str | os.PathLike) -> None:
    lines = [
        '# Dataset Readiness',
        '',
        f'- data: `{report["data_yaml"]}`',
        f'- ready_for_yolov8s: `{report["ready_for_yolov8s"]}`',
        f'- recommendation: {report["recommendation"]}',
        '',
        '## Problems',
        '',
    ]
    if report['problems']:
        lines.extend(f'- {item}' for item in report['problems'])
    else:
        lines.append('- none')
    lines.extend(['', '## Warnings', ''])
    if report['warnings']:
        lines.extend(f'- {item}' for item in report['warnings'])
    else:
        lines.append('- none')
    lines.extend(['', '## Class Counts', ''])
    for name, count in sorted(report['class_counts'].items(), key=lambda item: -item[1]):
        minimum = report['thresholds'].get(name)
        suffix = '' if minimum is None else f' / min {minimum}'
        lines.append(f'- `{name}`: {count}{suffix}')
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='Check YOLO dataset readiness for YOLOv8s.')
    parser.add_argument('data_yaml')
    parser.add_argument('--min-boxes', action='append', default=[],
                        help='Override threshold, e.g. cigarette_butt:2500. Can repeat.')
    parser.add_argument('--json', default='')
    parser.add_argument('--md', default='')
    args = parser.parse_args()

    thresholds = _parse_min_boxes(args.min_boxes)
    report = check_readiness(args.data_yaml, thresholds)
    root = Path(report['data_yaml']).resolve().parent
    json_path = Path(args.json) if args.json else root / 'readiness.json'
    md_path = Path(args.md) if args.md else root / 'readiness.md'

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    write_markdown(report, md_path)
    print(f'[OK] readiness json: {json_path}')
    print(f'[OK] readiness md:   {md_path}')
    if not report['ready_for_yolov8s']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
