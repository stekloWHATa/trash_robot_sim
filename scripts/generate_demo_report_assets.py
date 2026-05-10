#!/usr/bin/env python3
"""Generate diploma-ready plots/statistics from detector JSONL logs."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evaluate_detection_run import (
    dedupe_detections,
    evaluate,
    load_detections,
    load_ground_truth,
)


PALETTE = [
    '#2F80ED',
    '#27AE60',
    '#F2994A',
    '#EB5757',
    '#9B51E0',
    '#00A7A7',
    '#B26A00',
    '#4F4F4F',
]


def _fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return 'n/a'
    if isinstance(value, int):
        return str(value)
    return f'{value:.{digits}f}'


def _mean(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def _median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _svg_text(text: str) -> str:
    return html.escape(str(text), quote=True)


def _write_bar_svg(
    path: Path,
    title: str,
    labels: list[str],
    values: list[float],
    y_label: str,
    color: str = '#2F80ED',
) -> None:
    width, height = 1100, 650
    left, right, top, bottom = 120, 50, 90, 150
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_value = max(values) if values else 1.0
    max_value = max(max_value, 1.0)
    step = plot_w / max(len(values), 1)
    bar_w = min(86, step * 0.68)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="42" text-anchor="middle" font-family="DejaVu Sans, Arial" font-size="28" font-weight="700">{_svg_text(title)}</text>',
        f'<text x="28" y="{top + plot_h / 2:.1f}" transform="rotate(-90 28 {top + plot_h / 2:.1f})" text-anchor="middle" font-family="DejaVu Sans, Arial" font-size="18">{_svg_text(y_label)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333" stroke-width="2"/>',
    ]
    for tick in range(6):
        value = max_value * tick / 5
        y = top + plot_h - (value / max_value) * plot_h
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e6e6e6" stroke-width="1"/>')
        lines.append(f'<text x="{left - 12}" y="{y + 5:.1f}" text-anchor="end" font-family="DejaVu Sans, Arial" font-size="14" fill="#333">{_fmt(value, 2)}</text>')

    for idx, (label, value) in enumerate(zip(labels, values)):
        x = left + idx * step + (step - bar_w) / 2
        h = 0.0 if max_value <= 0 else (value / max_value) * plot_h
        y = top + plot_h - h
        lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}" rx="5"/>')
        lines.append(f'<text x="{x + bar_w / 2:.1f}" y="{max(y - 10, top + 18):.1f}" text-anchor="middle" font-family="DejaVu Sans, Arial" font-size="15" font-weight="700" fill="#222">{_fmt(value, 2)}</text>')
        lx = x + bar_w / 2
        ly = top + plot_h + 24
        lines.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="end" transform="rotate(-36 {lx:.1f} {ly:.1f})" font-family="DejaVu Sans, Arial" font-size="14" fill="#222">{_svg_text(label)}</text>')

    lines.append('</svg>')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _histogram(values: list[float], bins: int = 12) -> tuple[list[str], list[float]]:
    if not values:
        return [], []
    lo, hi = min(values), max(values)
    if math.isclose(lo, hi):
        return [f'{lo:.1f}'], [float(len(values))]
    width = (hi - lo) / bins
    counts = [0 for _ in range(bins)]
    for value in values:
        idx = min(bins - 1, int((value - lo) / width))
        counts[idx] += 1
    labels = [f'{lo + i * width:.1f}-{lo + (i + 1) * width:.1f}' for i in range(bins)]
    return labels, [float(count) for count in counts]


def _write_scatter_svg(
    path: Path,
    title: str,
    detections: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
) -> None:
    width, height = 1100, 720
    left, right, top, bottom = 110, 260, 90, 90
    plot_w = width - left - right
    plot_h = height - top - bottom

    xs = [float(det['x']) for det in detections] + [float(gt['x']) for gt in ground_truth]
    ys = [float(det['y']) for det in detections] + [float(gt['y']) for gt in ground_truth]
    if not xs:
        xs, ys = [0.0, 1.0], [0.0, 1.0]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    pad_x = max(0.5, (max_x - min_x) * 0.12)
    pad_y = max(0.5, (max_y - min_y) * 0.12)
    min_x -= pad_x
    max_x += pad_x
    min_y -= pad_y
    max_y += pad_y

    classes = sorted({det['class'] for det in detections} | {gt['class'] for gt in ground_truth})
    colors = {cls: PALETTE[idx % len(PALETTE)] for idx, cls in enumerate(classes)}

    def sx(x: float) -> float:
        return left + (x - min_x) / (max_x - min_x) * plot_w

    def sy(y: float) -> float:
        return top + plot_h - (y - min_y) / (max_y - min_y) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="42" text-anchor="middle" font-family="DejaVu Sans, Arial" font-size="28" font-weight="700">{_svg_text(title)}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fafafa" stroke="#333" stroke-width="2"/>',
    ]
    for tick in range(7):
        x = left + plot_w * tick / 6
        y = top + plot_h * tick / 6
        vx = min_x + (max_x - min_x) * tick / 6
        vy = max_y - (max_y - min_y) * tick / 6
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#e4e4e4"/>')
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e4e4e4"/>')
        lines.append(f'<text x="{x:.1f}" y="{top + plot_h + 28}" text-anchor="middle" font-family="DejaVu Sans, Arial" font-size="14">{vx:.1f}</text>')
        lines.append(f'<text x="{left - 12}" y="{y + 5:.1f}" text-anchor="end" font-family="DejaVu Sans, Arial" font-size="14">{vy:.1f}</text>')

    for gt in ground_truth:
        x, y = sx(float(gt['x'])), sy(float(gt['y']))
        color = colors.get(gt['class'], '#111')
        lines.append(f'<line x1="{x - 9:.1f}" y1="{y - 9:.1f}" x2="{x + 9:.1f}" y2="{y + 9:.1f}" stroke="{color}" stroke-width="4"/>')
        lines.append(f'<line x1="{x - 9:.1f}" y1="{y + 9:.1f}" x2="{x + 9:.1f}" y2="{y - 9:.1f}" stroke="{color}" stroke-width="4"/>')

    for det in detections:
        x, y = sx(float(det['x'])), sy(float(det['y']))
        color = colors.get(det['class'], '#555')
        conf = max(0.0, min(1.0, float(det.get('confidence', 0.0))))
        radius = 4.0 + 7.0 * conf
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{color}" fill-opacity="0.42" stroke="{color}" stroke-width="1.5"/>')

    legend_x = left + plot_w + 32
    lines.append(f'<text x="{legend_x}" y="{top}" font-family="DejaVu Sans, Arial" font-size="18" font-weight="700">Legend</text>')
    lines.append(f'<text x="{legend_x}" y="{top + 30}" font-family="DejaVu Sans, Arial" font-size="14">circle = detection</text>')
    lines.append(f'<text x="{legend_x}" y="{top + 52}" font-family="DejaVu Sans, Arial" font-size="14">cross = ground truth</text>')
    for idx, cls in enumerate(classes):
        y = top + 92 + idx * 30
        color = colors[cls]
        lines.append(f'<rect x="{legend_x}" y="{y - 14}" width="18" height="18" fill="{color}"/>')
        lines.append(f'<text x="{legend_x + 28}" y="{y}" font-family="DejaVu Sans, Arial" font-size="14">{_svg_text(cls)}</text>')
    lines.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 22}" text-anchor="middle" font-family="DejaVu Sans, Arial" font-size="16">world X, m</text>')
    lines.append(f'<text x="32" y="{top + plot_h / 2:.1f}" transform="rotate(-90 32 {top + plot_h / 2:.1f})" text-anchor="middle" font-family="DejaVu Sans, Arial" font-size="16">world Y, m</text>')
    lines.append('</svg>')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _class_statistics(detections: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for det in detections:
        grouped[det['class']].append(det)

    stats = {}
    for cls, items in sorted(grouped.items()):
        confs = [float(item.get('confidence', 0.0)) for item in items]
        depths = [
            float(item['raw']['depth_m'])
            for item in items
            if isinstance(item.get('raw'), dict) and item['raw'].get('depth_m') is not None
        ]
        stats[cls] = {
            'detections': len(items),
            'mean_confidence': _mean(confs),
            'median_confidence': _median(confs),
            'mean_depth_m': _mean(depths),
            'median_depth_m': _median(depths),
            'min_depth_m': min(depths) if depths else None,
            'max_depth_m': max(depths) if depths else None,
        }
    return stats


def _write_class_csv(path: Path, stats: dict[str, dict[str, Any]]) -> None:
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'class',
            'detections',
            'mean_confidence',
            'median_confidence',
            'mean_depth_m',
            'median_depth_m',
            'min_depth_m',
            'max_depth_m',
        ])
        for cls, row in sorted(stats.items()):
            writer.writerow([
                cls,
                row['detections'],
                row['mean_confidence'],
                row['median_confidence'],
                row['mean_depth_m'],
                row['median_depth_m'],
                row['min_depth_m'],
                row['max_depth_m'],
            ])


def _write_markdown(path: Path, summary: dict[str, Any], class_stats: dict[str, dict[str, Any]]) -> None:
    lines = [
        '# Detection Demo Report Assets',
        '',
        'This folder is generated from the detector JSONL log and Gazebo ground truth.',
        '',
        '## Summary',
        '',
        f"- detections_raw: {summary['detections_raw']}",
        f"- detections_deduped: {summary['detections_deduped']}",
        f"- classes_detected: {summary['classes_detected']}",
        f"- mean_confidence: {_fmt(summary['mean_confidence'])}",
        f"- median_confidence: {_fmt(summary['median_confidence'])}",
        f"- mean_inference_ms: {_fmt(summary['mean_inference_ms'])}",
        f"- median_inference_ms: {_fmt(summary['median_inference_ms'])}",
        f"- estimated_fps_from_log: {_fmt(summary['estimated_fps_from_log'])}",
        f"- source_counts: {summary['source_counts']}",
        '',
        '## Per Class',
        '',
        '| class | detections | mean conf | median conf | mean depth, m |',
        '| --- | ---: | ---: | ---: | ---: |',
    ]
    for cls, row in sorted(class_stats.items()):
        lines.append(
            f"| {cls} | {row['detections']} | {_fmt(row['mean_confidence'])} | "
            f"{_fmt(row['median_confidence'])} | {_fmt(row['mean_depth_m'])} |"
        )
    lines.extend([
        '',
        '## Generated Files',
        '',
        '- `summary.json`',
        '- `detections_by_class.csv`',
        '- `detections_by_class.svg`',
        '- `confidence_by_class.svg`',
        '- `depth_by_class.svg`',
        '- `world_scatter.svg`',
        '- `latency_histogram.svg`',
        '- `detections_timeline.svg`',
        '- `evaluation_snapshot.json`',
    ])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def generate_report_assets(
    detections_path: str | Path,
    ground_truth_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ground_truth, match_distance = load_ground_truth(ground_truth_path)
    detections = load_detections(detections_path)
    deduped = dedupe_detections(detections)
    evaluation = evaluate(ground_truth, detections, match_distance)

    class_stats = _class_statistics(detections)
    confs = [float(det.get('confidence', 0.0)) for det in detections]
    latencies = [
        float(det['inference_ms'])
        for det in detections
        if det.get('inference_ms') is not None
    ]
    timestamps = [
        float(det['timestamp_wall'])
        for det in detections
        if det.get('timestamp_wall') is not None
    ]
    frames = {
        det.get('frame_seq')
        for det in detections
        if det.get('frame_seq') is not None
    }
    duration = max(timestamps) - min(timestamps) if len(timestamps) >= 2 else None
    fps = len(frames) / duration if frames and duration and duration > 0 else None
    class_counts = Counter(det['class'] for det in detections)
    source_counts = Counter(
        det.get('raw', {}).get('source', 'yolo')
        for det in detections
    )

    summary = {
        'detections_log': str(detections_path),
        'ground_truth': str(ground_truth_path),
        'detections_raw': len(detections),
        'detections_deduped': len(deduped),
        'classes_detected': sorted(class_counts),
        'source_counts': dict(sorted(source_counts.items())),
        'mean_confidence': _mean(confs),
        'median_confidence': _median(confs),
        'mean_inference_ms': _mean(latencies),
        'median_inference_ms': _median(latencies),
        'max_inference_ms': max(latencies) if latencies else None,
        'estimated_fps_from_log': fps,
        'evaluation_metrics': evaluation.get('metrics', {}),
        'localization_error_m': evaluation.get('localization_error_m', {}),
    }

    (output_dir / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    (output_dir / 'evaluation_snapshot.json').write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    _write_class_csv(output_dir / 'detections_by_class.csv', class_stats)
    _write_markdown(output_dir / 'summary.md', summary, class_stats)

    labels = list(class_counts)
    counts = [float(class_counts[label]) for label in labels]
    _write_bar_svg(
        output_dir / 'detections_by_class.svg',
        'Detections by Class',
        labels,
        counts,
        'detections',
        '#2F80ED',
    )

    confidence_labels = list(class_stats)
    confidence_values = [
        float(class_stats[label]['mean_confidence'] or 0.0)
        for label in confidence_labels
    ]
    _write_bar_svg(
        output_dir / 'confidence_by_class.svg',
        'Mean Confidence by Class',
        confidence_labels,
        confidence_values,
        'confidence',
        '#27AE60',
    )

    depth_labels = [
        label for label, row in class_stats.items()
        if row.get('mean_depth_m') is not None
    ]
    depth_values = [float(class_stats[label]['mean_depth_m']) for label in depth_labels]
    _write_bar_svg(
        output_dir / 'depth_by_class.svg',
        'Mean RGBD Depth by Class',
        depth_labels,
        depth_values,
        'depth, m',
        '#F2994A',
    )

    hist_labels, hist_values = _histogram(latencies, bins=12)
    _write_bar_svg(
        output_dir / 'latency_histogram.svg',
        'Detector Inference Latency Histogram',
        hist_labels,
        hist_values,
        'frames',
        '#9B51E0',
    )

    if timestamps:
        start = min(timestamps)
        buckets = Counter(int(float(ts) - start) for ts in timestamps)
        timeline_labels = [str(sec) for sec in range(0, max(buckets) + 1)]
        timeline_values = [float(buckets.get(sec, 0)) for sec in range(0, max(buckets) + 1)]
    else:
        timeline_labels, timeline_values = [], []
    _write_bar_svg(
        output_dir / 'detections_timeline.svg',
        'Detections Timeline',
        timeline_labels,
        timeline_values,
        'detections / second',
        '#EB5757',
    )

    _write_scatter_svg(
        output_dir / 'world_scatter.svg',
        'World Localization Scatter',
        detections,
        ground_truth,
    )

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate SVG/CSV/JSON report assets from detector logs.')
    parser.add_argument(
        '--detections',
        default='/tmp/trash_detections.jsonl',
        help='JSONL log produced by trash_detector.',
    )
    parser.add_argument(
        '--ground-truth',
        default='config/trash_ground_truth.yaml',
        help='Gazebo object ground truth YAML.',
    )
    parser.add_argument(
        '--output-dir',
        default='reports/detection_demo/latest',
        help='Directory for generated report assets.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = generate_report_assets(
        detections_path=args.detections,
        ground_truth_path=args.ground_truth,
        output_dir=args.output_dir,
    )
    print(f"Wrote report assets to {args.output_dir}")
    print(
        'detections={detections_raw}, classes={classes}'.format(
            detections_raw=summary['detections_raw'],
            classes=', '.join(summary['classes_detected']),
        )
    )


if __name__ == '__main__':
    main()
