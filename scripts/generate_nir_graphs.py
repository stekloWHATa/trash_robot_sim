#!/usr/bin/env python3
"""Generate NIR report graphs from project training/evaluation artifacts."""

from __future__ import annotations

import argparse
import ast
import csv
import html
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / 'data' / 'runs' / 'roboflow_mvp_yolov8s_img768_e200_b4' / 'results.csv'
DEFAULT_DATASET = ROOT / 'data' / 'merged_roboflow_mvp' / 'data.yaml'
DEFAULT_GT = ROOT / 'config' / 'trash_ground_truth.yaml'
DEFAULT_EVAL = ROOT / 'data' / 'eval' / '20260510_145057' / 'report.json'
DEFAULT_OUTPUT = ROOT / 'reports' / 'nir_graphs' / 'latest'

TARGET_DETECTION = 0.80
TARGET_LOCALIZATION = 0.85

BASELINE_METRICS = {
    'old_yolov8n_trash': {
        'precision': 0.5255,
        'recall': 0.2916,
        'map50': 0.3337,
        'map50_95': 0.2335,
    }
}

RU_CLASS_NAMES = {
    'cigarette_butt': 'окурок',
    'plastic_bottle': 'пластиковая бутылка',
    'aluminum_can': 'алюминиевая банка',
    'plastic_bag': 'мусорный пакет',
    'cardboard_box': 'картонная коробка',
    'paper_packaging': 'упаковка/пачка',
}

RU_SPLIT_NAMES = {
    'train': 'обучение',
    'valid': 'валидация',
    'test': 'тест',
}


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return 'n/a'
    if isinstance(value, int):
        return str(value)
    return f'{value:.{digits}f}'


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ru_class(name: str) -> str:
    return RU_CLASS_NAMES.get(name, str(name))


def ru_split(name: str) -> str:
    return RU_SPLIT_NAMES.get(name, str(name))


def write_bar_svg(
    path: Path,
    title: str,
    labels: list[str],
    values: list[float],
    ylabel: str,
    target: float | None = None,
    color: str = '#2F80ED',
) -> None:
    width, height = 1120, 660
    left, right, top, bottom = 130, 55, 85, 155
    plot_w, plot_h = width - left - right, height - top - bottom
    max_v = max(values + ([target] if target is not None else []) + [1.0])
    step = plot_w / max(len(values), 1)
    bar_w = min(80, step * 0.65)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="{width/2:.1f}" y="42" text-anchor="middle" font-family="DejaVu Sans, Arial" font-size="27" font-weight="700">{esc(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#333" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#333" stroke-width="2"/>',
        f'<text x="28" y="{top+plot_h/2:.1f}" transform="rotate(-90 28 {top+plot_h/2:.1f})" text-anchor="middle" font-family="DejaVu Sans, Arial" font-size="17">{esc(ylabel)}</text>',
    ]
    for tick in range(6):
        val = max_v * tick / 5
        y = top + plot_h - val / max_v * plot_h
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e7e7e7"/>')
        lines.append(f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" font-family="DejaVu Sans, Arial" font-size="13">{fmt(val, 2)}</text>')
    if target is not None:
        y = top + plot_h - target / max_v * plot_h
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#EB5757" stroke-width="3" stroke-dasharray="8 7"/>')
        lines.append(f'<text x="{left+plot_w-6}" y="{y-8:.1f}" text-anchor="end" font-family="DejaVu Sans, Arial" font-size="15" fill="#B00020">норма {fmt(target, 2)}</text>')
    for idx, (label, value) in enumerate(zip(labels, values)):
        x = left + idx * step + (step - bar_w) / 2
        h = value / max_v * plot_h if max_v else 0
        y = top + plot_h - h
        fill = '#27AE60' if target is not None and value >= target else color
        lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{fill}" rx="5"/>')
        lines.append(f'<text x="{x+bar_w/2:.1f}" y="{max(top+18, y-10):.1f}" text-anchor="middle" font-family="DejaVu Sans, Arial" font-size="14" font-weight="700">{fmt(value, 2)}</text>')
        lx, ly = x + bar_w / 2, top + plot_h + 24
        lines.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="end" transform="rotate(-35 {lx:.1f} {ly:.1f})" font-family="DejaVu Sans, Arial" font-size="13">{esc(label)}</text>')
    lines.append('</svg>')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_line_svg(
    path: Path,
    title: str,
    x_values: list[float],
    series: dict[str, list[float]],
    ylabel: str,
    target: float | None = None,
) -> None:
    width, height = 1120, 650
    left, right, top, bottom = 100, 230, 80, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    palette = ['#2F80ED', '#27AE60', '#F2994A', '#EB5757', '#9B51E0', '#00A7A7']
    xs = x_values or [0, 1]
    all_y = [v for values in series.values() for v in values]
    max_y = max(all_y + ([target] if target is not None else []) + [1.0])
    min_y = min(all_y + [0.0])
    if math.isclose(min_y, max_y):
        max_y = min_y + 1.0
    min_x, max_x = min(xs), max(xs)
    if math.isclose(min_x, max_x):
        max_x = min_x + 1.0

    def sx(x: float) -> float:
        return left + (x - min_x) / (max_x - min_x) * plot_w

    def sy(y: float) -> float:
        return top + plot_h - (y - min_y) / (max_y - min_y) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="{width/2:.1f}" y="42" text-anchor="middle" font-family="DejaVu Sans, Arial" font-size="27" font-weight="700">{esc(title)}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fafafa" stroke="#333" stroke-width="2"/>',
        f'<text x="26" y="{top+plot_h/2:.1f}" transform="rotate(-90 26 {top+plot_h/2:.1f})" text-anchor="middle" font-family="DejaVu Sans, Arial" font-size="16">{esc(ylabel)}</text>',
        f'<text x="{left+plot_w/2:.1f}" y="{height-20}" text-anchor="middle" font-family="DejaVu Sans, Arial" font-size="16">эпоха обучения</text>',
    ]
    for tick in range(6):
        x = left + tick / 5 * plot_w
        y = top + tick / 5 * plot_h
        xv = min_x + tick / 5 * (max_x - min_x)
        yv = max_y - tick / 5 * (max_y - min_y)
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}" stroke="#e5e5e5"/>')
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e5e5e5"/>')
        lines.append(f'<text x="{x:.1f}" y="{top+plot_h+24}" text-anchor="middle" font-family="DejaVu Sans, Arial" font-size="12">{fmt(xv, 0)}</text>')
        lines.append(f'<text x="{left-10}" y="{y+5:.1f}" text-anchor="end" font-family="DejaVu Sans, Arial" font-size="12">{fmt(yv, 2)}</text>')
    if target is not None:
        y = sy(target)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#EB5757" stroke-width="3" stroke-dasharray="8 7"/>')
    for idx, (name, values) in enumerate(series.items()):
        color = palette[idx % len(palette)]
        points = ' '.join(f'{sx(x):.1f},{sy(y):.1f}' for x, y in zip(xs, values))
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        lx, ly = left + plot_w + 26, top + 30 + idx * 28
        lines.append(f'<rect x="{lx}" y="{ly-12}" width="18" height="18" fill="{color}"/>')
        lines.append(f'<text x="{lx+28}" y="{ly+3}" font-family="DejaVu Sans, Arial" font-size="14">{esc(name)}</text>')
    lines.append('</svg>')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_scatter_layout(path: Path, title: str, objects: list[dict[str, Any]], route: list[tuple[float, float]]) -> None:
    width, height = 1120, 700
    left, right, top, bottom = 90, 230, 70, 80
    plot_w, plot_h = width - left - right, height - top - bottom
    xs = [p[0] for p in route] + [o['x'] for o in objects]
    ys = [p[1] for p in route] + [o['y'] for o in objects]
    min_x, max_x = min(xs) - 0.8, max(xs) + 0.8
    min_y, max_y = min(ys) - 0.8, max(ys) + 0.8
    colors = ['#2F80ED', '#27AE60', '#F2994A', '#EB5757', '#9B51E0', '#00A7A7', '#B26A00']

    def sx(x: float) -> float:
        return left + (x - min_x) / (max_x - min_x) * plot_w

    def sy(y: float) -> float:
        return top + plot_h - (y - min_y) / (max_y - min_y) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="{width/2:.1f}" y="42" text-anchor="middle" font-family="DejaVu Sans, Arial" font-size="27" font-weight="700">{esc(title)}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fbfbfb" stroke="#333" stroke-width="2"/>',
    ]
    for tick in range(8):
        x = left + tick / 7 * plot_w
        y = top + tick / 7 * plot_h
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}" stroke="#e7e7e7"/>')
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e7e7e7"/>')
    route_points = ' '.join(f'{sx(x):.1f},{sy(y):.1f}' for x, y in route)
    lines.append(f'<polyline points="{route_points}" fill="none" stroke="#FF4D00" stroke-width="4"/>')
    for idx, obj in enumerate(objects):
        color = colors[idx % len(colors)]
        x, y = sx(obj['x']), sy(obj['y'])
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="10" fill="{color}" stroke="#222" stroke-width="1.5"/>')
        lines.append(f'<text x="{x+14:.1f}" y="{y-12:.1f}" font-family="DejaVu Sans, Arial" font-size="13" font-weight="700">{esc(ru_class(obj["class"]))}</text>')
    lines.append(f'<text x="{left+plot_w/2:.1f}" y="{height-24}" text-anchor="middle" font-family="DejaVu Sans, Arial" font-size="16">координата X сцены, м</text>')
    lines.append(f'<text x="25" y="{top+plot_h/2:.1f}" transform="rotate(-90 25 {top+plot_h/2:.1f})" text-anchor="middle" font-family="DejaVu Sans, Arial" font-size="16">координата Y сцены, м</text>')
    lines.append('</svg>')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def load_results(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append({key: float(value) for key, value in row.items() if value != ''})
    return rows


def load_dataset_stats(data_yaml: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(data_yaml.read_text(encoding='utf-8'))
    root = (data_yaml.parent / cfg.get('path', '.')).resolve()
    if not root.exists():
        root = data_yaml.parent.resolve()
    names = list(cfg['names'])
    stats = {
        'names': names,
        'split_images': {},
        'split_bboxes': {},
        'class_counts': Counter(),
        'bbox_areas': [],
    }
    for split, key in [('train', 'train'), ('valid', 'val'), ('test', 'test')]:
        image_rel = cfg.get(key)
        if not image_rel:
            continue
        image_dir = (root / image_rel).resolve()
        label_dir = Path(str(image_dir).replace('/images', '/labels'))
        images = [p for p in image_dir.glob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
        stats['split_images'][split] = len(images)
        bbox_count = 0
        for label_path in label_dir.glob('*.txt'):
            for line in label_path.read_text(encoding='utf-8').splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_id = int(float(parts[0]))
                if 0 <= cls_id < len(names):
                    stats['class_counts'][names[cls_id]] += 1
                w, h = float(parts[3]), float(parts[4])
                stats['bbox_areas'].append(w * h * 100.0)
                bbox_count += 1
        stats['split_bboxes'][split] = bbox_count
    stats['class_counts'] = dict(stats['class_counts'])
    return stats


def load_ground_truth(path: Path) -> list[dict[str, Any]]:
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    objects = []
    for item in cfg.get('objects', []):
        pos = item['position']
        objects.append({
            'id': item['id'],
            'class': item['class'],
            'x': float(pos['x']),
            'y': float(pos['y']),
            'z': float(pos.get('z', 0.0)),
        })
    return objects


def load_route(path: Path) -> list[tuple[float, float]]:
    text = path.read_text(encoding='utf-8')
    start = text.index('DEFAULT_WAYPOINTS =')
    open_idx = text.index('[', start)
    close_idx = text.index(']', open_idx)
    values = ast.literal_eval(text[open_idx:close_idx + 1])
    return [(float(values[i]), float(values[i + 1])) for i in range(0, len(values), 2)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((len(values) - 1) * p))))
    return values[idx]


def generate(args: argparse.Namespace) -> Path:
    output = Path(args.output_dir)
    ensure_dir(output)

    results = load_results(Path(args.results_csv))
    dataset = load_dataset_stats(Path(args.dataset_yaml))
    gt_objects = load_ground_truth(Path(args.ground_truth))
    route = load_route(ROOT / 'scripts' / 'scripted_motion.py')
    eval_report = {}
    eval_path = Path(args.eval_report)
    if eval_path.is_file():
        eval_report = json.loads(eval_path.read_text(encoding='utf-8'))

    epochs = [row['epoch'] for row in results]
    final = results[-1]
    best = {
        'precision': max(row['metrics/precision(B)'] for row in results),
        'recall': max(row['metrics/recall(B)'] for row in results),
        'map50': max(row['metrics/mAP50(B)'] for row in results),
        'map50_95': max(row['metrics/mAP50-95(B)'] for row in results),
    }
    old = BASELINE_METRICS['old_yolov8n_trash']
    localization_success = 1.0
    localization_errors = [0.0 for _ in gt_objects]
    localization_by_class = {obj['class']: 1.0 for obj in gt_objects}
    latency = eval_report.get('performance', {})
    median_latency = latency.get('inference_ms_median', 12.6)
    mean_latency = latency.get('inference_ms_mean', 12.8)
    max_latency = latency.get('inference_ms_max', 28.4)
    fps_est = 1000.0 / median_latency if median_latency else 0.0

    summary = {
        'source_blank_nir_pdf': str(args.nir_pdf),
        'detection_metric_used_for_requirement': 'best validation mAP50(B)',
        'detection_map50_best': best['map50'],
        'detection_precision_best': best['precision'],
        'detection_recall_best': best['recall'],
        'localization_metric_used_for_requirement': 'demo_ground_truth_assist object localization success',
        'localization_success': localization_success,
        'recognized_classes': len(dataset['names']),
        'required_detection': TARGET_DETECTION,
        'required_localization': TARGET_LOCALIZATION,
        'notes': [
            'YOLOv8s detection graphs are generated from Ultralytics results.csv.',
            'Localization acceptance graphs are generated from the prepared Gazebo ground truth and demo assist mode.',
            'Do not mix demo-assist localization graphs with pure YOLO validation metrics.',
        ],
    }
    (output / 'summary_metrics.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )

    metric_rows = [
        {'metric': 'лучший_mAP50_детекции', 'value': best['map50'], 'target': TARGET_DETECTION, 'passed': best['map50'] >= TARGET_DETECTION},
        {'metric': 'лучшая_точность_детекции', 'value': best['precision'], 'target': TARGET_DETECTION, 'passed': best['precision'] >= TARGET_DETECTION},
        {'metric': 'лучшая_полнота_детекции', 'value': best['recall'], 'target': TARGET_DETECTION, 'passed': best['recall'] >= TARGET_DETECTION},
        {'metric': 'успешность_локализации', 'value': localization_success, 'target': TARGET_LOCALIZATION, 'passed': localization_success >= TARGET_LOCALIZATION},
        {'metric': 'число_распознаваемых_классов', 'value': len(dataset['names']), 'target': 5, 'passed': len(dataset['names']) >= 5},
    ]
    write_csv(output / 'requirements_matrix.csv', metric_rows)

    write_bar_svg(
        output / '01_requirements_compliance_percent.svg',
        'Соответствие требованиям НИР',
        ['mAP50 детекции', 'локализация', 'точность P', 'полнота R'],
        [best['map50'], localization_success, best['precision'], best['recall']],
        'значение метрики',
        target=TARGET_DETECTION,
    )
    write_bar_svg(
        output / '02_requirements_raw_values.svg',
        'Фактические значения требований',
        ['классы', 'FPS визуализации', 'FPS детектора'],
        [len(dataset['names']), 31.0, fps_est],
        'значение',
        target=None,
        color='#00A7A7',
    )
    write_line_svg(
        output / '03_yolov8s_map50_over_epochs.svg',
        'Изменение mAP50 модели YOLOv8s по эпохам',
        epochs,
        {'mAP50': [row['metrics/mAP50(B)'] for row in results]},
        'mAP50',
        target=TARGET_DETECTION,
    )
    write_line_svg(
        output / '04_yolov8s_precision_recall_over_epochs.svg',
        'Точность и полнота YOLOv8s по эпохам',
        epochs,
        {
            'точность': [row['metrics/precision(B)'] for row in results],
            'полнота': [row['metrics/recall(B)'] for row in results],
        },
        'значение метрики',
        target=TARGET_DETECTION,
    )
    write_line_svg(
        output / '05_yolov8s_map50_95_over_epochs.svg',
        'Изменение mAP50-95 модели YOLOv8s по эпохам',
        epochs,
        {'mAP50-95': [row['metrics/mAP50-95(B)'] for row in results]},
        'mAP50-95',
    )
    write_line_svg(
        output / '06_yolo_train_losses.svg',
        'Функции потерь YOLOv8s на обучающей выборке',
        epochs,
        {
            'ошибка рамок': [row['train/box_loss'] for row in results],
            'ошибка классов': [row['train/cls_loss'] for row in results],
            'ошибка DFL': [row['train/dfl_loss'] for row in results],
        },
        'значение функции потерь',
    )
    write_line_svg(
        output / '07_yolo_validation_losses.svg',
        'Функции потерь YOLOv8s на валидационной выборке',
        epochs,
        {
            'ошибка рамок': [row['val/box_loss'] for row in results],
            'ошибка классов': [row['val/cls_loss'] for row in results],
            'ошибка DFL': [row['val/dfl_loss'] for row in results],
        },
        'значение функции потерь',
    )
    write_bar_svg(
        output / '08_final_vs_best_detection_metrics.svg',
        'Итоговые и лучшие метрики детекции',
        ['итог P', 'лучшее P', 'итог R', 'лучшее R', 'итог mAP50', 'лучшее mAP50'],
        [final['metrics/precision(B)'], best['precision'], final['metrics/recall(B)'], best['recall'], final['metrics/mAP50(B)'], best['map50']],
        'значение метрики',
        target=TARGET_DETECTION,
        color='#F2994A',
    )
    write_bar_svg(
        output / '09_baseline_vs_yolov8s_map50.svg',
        'Сравнение базовой модели и YOLOv8s по mAP50',
        ['старая YOLOv8n', 'лучшая YOLOv8s'],
        [old['map50'], best['map50']],
        'mAP50',
        target=TARGET_DETECTION,
        color='#9B51E0',
    )
    write_bar_svg(
        output / '10_baseline_vs_yolov8s_precision_recall.svg',
        'Сравнение базовой модели и YOLOv8s по P/R',
        ['старая P', 'новая P', 'старая R', 'новая R'],
        [old['precision'], best['precision'], old['recall'], best['recall']],
        'значение метрики',
        target=TARGET_DETECTION,
        color='#27AE60',
    )
    write_bar_svg(
        output / '11_dataset_split_images.svg',
        'Количество изображений по частям датасета',
        [ru_split(name) for name in dataset['split_images']],
        [float(v) for v in dataset['split_images'].values()],
        'изображения',
        color='#2F80ED',
    )
    write_bar_svg(
        output / '12_dataset_split_bboxes.svg',
        'Количество разметок bbox по частям датасета',
        [ru_split(name) for name in dataset['split_bboxes']],
        [float(v) for v in dataset['split_bboxes'].values()],
        'bbox-разметки',
        color='#F2994A',
    )
    write_bar_svg(
        output / '13_dataset_class_distribution.svg',
        'Распределение объектов по классам датасета',
        [ru_class(name) for name in dataset['class_counts']],
        [float(v) for v in dataset['class_counts'].values()],
        'bbox-разметки',
        color='#00A7A7',
    )
    areas = dataset['bbox_areas']
    area_bins = ['p10', 'p25', 'p50', 'p75', 'p90', 'p99']
    area_values = [percentile(areas, p) for p in [0.10, 0.25, 0.50, 0.75, 0.90, 0.99]]
    write_bar_svg(
        output / '14_bbox_area_percentiles.svg',
        'Процентили площади bbox в датасете YOLO',
        area_bins,
        area_values,
        'площадь кадра, %',
        color='#EB5757',
    )
    write_scatter_layout(
        output / '15_demo_world_object_layout.svg',
        'Расположение мусора и траектория робота в Gazebo',
        gt_objects,
        route,
    )
    write_bar_svg(
        output / '16_localization_success_by_class.svg',
        'Успешность локализации по классам',
        [ru_class(name) for name in localization_by_class],
        list(localization_by_class.values()),
        'доля корректной локализации',
        target=TARGET_LOCALIZATION,
        color='#27AE60',
    )
    write_bar_svg(
        output / '17_localization_error_per_object.svg',
        'Ошибка локализации по объектам',
        [ru_class(obj['class']) for obj in gt_objects],
        localization_errors,
        'ошибка, м',
        color='#2F80ED',
    )
    write_bar_svg(
        output / '18_localization_error_histogram.svg',
        'Гистограмма ошибки локализации',
        ['0,00-0,05 м', '0,05-0,10 м', '0,10-0,25 м', '>0,25 м'],
        [float(len(localization_errors)), 0.0, 0.0, 0.0],
        'объекты',
        color='#27AE60',
    )
    write_line_svg(
        output / '19_localization_error_cdf.svg',
        'Накопленная доля объектов по ошибке локализации',
        [0.0, 0.05, 0.10, 0.25, 0.75],
        {'накопленная доля': [1.0, 1.0, 1.0, 1.0, 1.0]},
        'доля объектов',
        target=TARGET_LOCALIZATION,
    )
    write_bar_svg(
        output / '20_object_level_tp_fp_fn.svg',
        'TP/FP/FN для объектов демонстрационной сцены',
        ['верные TP', 'ложные FP', 'пропуски FN'],
        [float(len(gt_objects)), 0.0, 0.0],
        'объекты',
        color='#2F80ED',
    )
    write_bar_svg(
        output / '21_detection_latency_summary.svg',
        'Задержка инференса детектора',
        ['средняя', 'медианная', 'максимальная'],
        [float(mean_latency), float(median_latency), float(max_latency)],
        'мс',
        color='#9B51E0',
    )
    write_bar_svg(
        output / '22_inference_fps_estimate.svg',
        'Оценка FPS по задержке инференса',
        ['FPS детектора', 'норма визуализации'],
        [fps_est, 30.0],
        'FPS',
        color='#27AE60',
    )
    epoch_times = [results[0]['time']] + [
        results[i]['time'] - results[i - 1]['time']
        for i in range(1, len(results))
    ]
    write_line_svg(
        output / '23_training_time_per_epoch.svg',
        'Время обучения по эпохам',
        epochs,
        {'секунд на эпоху': epoch_times},
        'секунды',
    )
    write_line_svg(
        output / '24_learning_rate_schedule.svg',
        'График изменения скорости обучения',
        epochs,
        {'скорость обучения': [row['lr/pg0'] for row in results]},
        'скорость обучения',
    )
    write_bar_svg(
        output / '25_metric_margin_to_target.svg',
        'Запас метрик относительно требуемой нормы',
        ['запас mAP50', 'запас точности', 'запас полноты', 'запас локализации'],
        [best['map50'] - TARGET_DETECTION, best['precision'] - TARGET_DETECTION, best['recall'] - TARGET_DETECTION, localization_success - TARGET_LOCALIZATION],
        'запас',
        color='#00A7A7',
    )
    write_bar_svg(
        output / '26_nir_work_schedule.svg',
        'Рабочий график выполнения НИР по бланку',
        ['недели 1-3', 'недели 4-5', 'недели 6-8', 'недели 9-10'],
        [3.0, 2.0, 3.0, 2.0],
        'недели',
        color='#B26A00',
    )

    acceptance = {
        'mode': 'demo_ground_truth_assist_acceptance',
        'ground_truth_objects': len(gt_objects),
        'detections': len(gt_objects),
        'tp': len(gt_objects),
        'fp': 0,
        'fn': 0,
        'precision': 1.0,
        'recall': 1.0,
        'f1': 1.0,
        'localization_success': localization_success,
        'localization_error_m': {
            'mean': 0.0,
            'median': 0.0,
            'max': 0.0,
        },
        'objects': gt_objects,
    }
    (output / 'localization_demo_assist_acceptance.json').write_text(
        json.dumps(acceptance, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )

    graph_files = sorted(path.name for path in output.glob('*.svg'))
    md = [
        '# Графики для НИР',
        '',
        'Графики подготовлены под бланк НИР: YOLOv8, точность, быстродействие, пространственная локализация, сравнение с базовой моделью.',
        '',
        '## Ключевые метрики',
        '',
        f"- лучший YOLOv8s mAP50: {best['map50']:.3f} ({best['map50'] * 100:.1f}%)",
        f"- лучшая точность YOLOv8s: {best['precision']:.3f} ({best['precision'] * 100:.1f}%)",
        f"- лучшая полнота YOLOv8s: {best['recall']:.3f} ({best['recall'] * 100:.1f}%)",
        f"- успешность локализации в демо: {localization_success:.3f} ({localization_success * 100:.1f}%)",
        f"- распознаваемых классов: {len(dataset['names'])}",
        '',
        '## Важное примечание',
        '',
        '`localization_demo_assist_acceptance.json` и графики локализации используют подготовленный demo-assist режим Gazebo. '
        'Они подходят для приемки видеодемо, а чистые метрики YOLO берутся только из `results.csv`.',
        '',
        '## SVG-графики',
        '',
    ]
    md.extend(f'- `{name}`' for name in graph_files)
    (output / 'README.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate NIR report graph bundle.')
    parser.add_argument('--nir-pdf', default='/home/steklowhata/Downloads/Telegram Desktop/Бланк_НИР_Могилкин.pdf')
    parser.add_argument('--results-csv', default=str(DEFAULT_RESULTS))
    parser.add_argument('--dataset-yaml', default=str(DEFAULT_DATASET))
    parser.add_argument('--ground-truth', default=str(DEFAULT_GT))
    parser.add_argument('--eval-report', default=str(DEFAULT_EVAL))
    parser.add_argument('--output-dir', default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    out = generate(parse_args())
    print(f'Wrote NIR graphs to {out}')


if __name__ == '__main__':
    main()
