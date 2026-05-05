#!/usr/bin/env python3
"""Evaluate a scripted Gazebo detection/localization demo run."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


CLASS_ALIASES = {
    'bottle': 'plastic_bottle',
    'can': 'aluminum_can',
    'can_cup': 'aluminum_can',
    'cup': 'aluminum_can',
    'cardboard': 'cardboard_box',
    'cardboard_paper': 'cardboard_box',
    'paper': 'paper_packaging',
}


def normalize_class(name: str) -> str:
    key = str(name).strip().lower().replace(' ', '_').replace('-', '_')
    return CLASS_ALIASES.get(key, key)


def load_ground_truth(path: str | Path) -> tuple[list[dict[str, Any]], float]:
    with Path(path).open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}

    match_distance = float(cfg.get('match_distance_m', 0.75))
    objects = []
    for item in cfg.get('objects', []):
        pos = item.get('position', {})
        objects.append({
            'id': str(item['id']),
            'class': normalize_class(item['class']),
            'x': float(pos['x']),
            'y': float(pos['y']),
            'z': float(pos.get('z', 0.0)),
        })
    return objects, match_distance


def _world_xy(record: dict[str, Any]) -> tuple[float, float] | None:
    world = record.get('world')
    if isinstance(world, dict) and world.get('x') is not None and world.get('y') is not None:
        return float(world['x']), float(world['y'])
    if record.get('world_x') is not None and record.get('world_y') is not None:
        return float(record['world_x']), float(record['world_y'])
    return None


def load_detections(path: str | Path) -> list[dict[str, Any]]:
    detections = []
    with Path(path).open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Bad JSONL line {line_no}: {exc}') from exc

            xy = _world_xy(raw)
            if xy is None:
                continue
            cls = raw.get('class') or raw.get('category') or raw.get('label')
            if cls is None:
                continue
            detections.append({
                'object_id': raw.get('object_id'),
                'class': normalize_class(cls),
                'label': raw.get('label', cls),
                'confidence': float(raw.get('confidence', raw.get('conf', 0.0))),
                'x': float(xy[0]),
                'y': float(xy[1]),
                'frame_seq': raw.get('frame_seq'),
                'timestamp_wall': raw.get('timestamp_wall'),
                'timestamp_ros': raw.get('timestamp_ros'),
                'inference_ms': raw.get('inference_ms'),
                'raw': raw,
            })
    return detections


def dedupe_detections(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one best record per detector object_id; keep id-less records as-is."""
    best_by_id: dict[Any, dict[str, Any]] = {}
    without_id: list[dict[str, Any]] = []
    for det in detections:
        object_id = det.get('object_id')
        if object_id is None:
            without_id.append(det)
            continue
        current = best_by_id.get(object_id)
        if current is None or det['confidence'] > current['confidence']:
            best_by_id[object_id] = det
    return list(best_by_id.values()) + without_id


def _distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    return math.hypot(float(a['x']) - float(b['x']), float(a['y']) - float(b['y']))


def match_detections(
    ground_truth: list[dict[str, Any]],
    detections: list[dict[str, Any]],
    match_distance: float,
) -> tuple[list[dict[str, Any]], set[int], set[int]]:
    pairs = []
    for gt_idx, gt in enumerate(ground_truth):
        for det_idx, det in enumerate(detections):
            if gt['class'] != det['class']:
                continue
            dist = _distance(gt, det)
            if dist <= match_distance:
                pairs.append((dist, gt_idx, det_idx))

    matched_gt: set[int] = set()
    matched_det: set[int] = set()
    matches = []
    for dist, gt_idx, det_idx in sorted(pairs, key=lambda item: item[0]):
        if gt_idx in matched_gt or det_idx in matched_det:
            continue
        matched_gt.add(gt_idx)
        matched_det.add(det_idx)
        matches.append({
            'gt_id': ground_truth[gt_idx]['id'],
            'det_object_id': detections[det_idx].get('object_id'),
            'class': ground_truth[gt_idx]['class'],
            'distance_m': dist,
            'confidence': detections[det_idx]['confidence'],
        })
    return matches, matched_gt, matched_det


def _safe_mean(values: list[float]) -> float | None:
    return None if not values else float(statistics.mean(values))


def _safe_median(values: list[float]) -> float | None:
    return None if not values else float(statistics.median(values))


def _safe_max(values: list[float]) -> float | None:
    return None if not values else float(max(values))


def evaluate(
    ground_truth: list[dict[str, Any]],
    detections: list[dict[str, Any]],
    match_distance: float,
) -> dict[str, Any]:
    raw_detection_count = len(detections)
    raw_for_perf = list(detections)
    detections = dedupe_detections(detections)
    matches, matched_gt, matched_det = match_detections(
        ground_truth, detections, match_distance)

    classes = sorted({obj['class'] for obj in ground_truth} | {det['class'] for det in detections})
    per_class: dict[str, dict[str, float | int | None]] = {}
    for cls in classes:
        gt_ids = {idx for idx, gt in enumerate(ground_truth) if gt['class'] == cls}
        det_ids = {idx for idx, det in enumerate(detections) if det['class'] == cls}
        tp = len(gt_ids & matched_gt)
        fp = len(det_ids - matched_det)
        fn = len(gt_ids - matched_gt)
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and (precision + recall) > 0.0
            else None
        )
        per_class[cls] = {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    errors = [float(item['distance_m']) for item in matches]
    latencies = [
        float(det['inference_ms'])
        for det in raw_for_perf
        if det.get('inference_ms') is not None
    ]
    timestamps = [
        float(det['timestamp_wall'])
        for det in raw_for_perf
        if det.get('timestamp_wall') is not None
    ]
    frame_ids = {
        det.get('frame_seq')
        for det in raw_for_perf
        if det.get('frame_seq') is not None
    }
    duration = max(timestamps) - min(timestamps) if len(timestamps) >= 2 else None
    fps = len(frame_ids) / duration if frame_ids and duration and duration > 0.0 else None

    total_tp = len(matches)
    total_fp = len(detections) - len(matched_det)
    total_fn = len(ground_truth) - len(matched_gt)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else None
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else None
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0.0
        else None
    )

    return {
        'match_distance_m': match_distance,
        'counts': {
            'ground_truth': len(ground_truth),
            'detections_raw': raw_detection_count,
            'detections_evaluated': len(detections),
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn,
        },
        'metrics': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
        },
        'per_class': per_class,
        'localization_error_m': {
            'mean': _safe_mean(errors),
            'median': _safe_median(errors),
            'max': _safe_max(errors),
        },
        'performance': {
            'fps_estimate_from_log': fps,
            'logged_frames': len(frame_ids) if frame_ids else None,
            'inference_ms_mean': _safe_mean(latencies),
            'inference_ms_median': _safe_median(latencies),
            'inference_ms_max': _safe_max(latencies),
        },
        'matches': matches,
        'unmatched_ground_truth': [
            ground_truth[idx] for idx in range(len(ground_truth)) if idx not in matched_gt
        ],
        'unmatched_detections': [
            {
                'object_id': detections[idx].get('object_id'),
                'class': detections[idx]['class'],
                'confidence': detections[idx]['confidence'],
                'x': detections[idx]['x'],
                'y': detections[idx]['y'],
            }
            for idx in range(len(detections)) if idx not in matched_det
        ],
    }


def write_markdown(report: dict[str, Any], path: str | Path) -> None:
    lines = [
        '# Detection Demo Evaluation',
        '',
        f"- match_distance_m: {report['match_distance_m']}",
        f"- ground_truth: {report['counts']['ground_truth']}",
        f"- detections: {report['counts']['detections_raw']}",
        f"- TP/FP/FN: {report['counts']['tp']}/{report['counts']['fp']}/{report['counts']['fn']}",
        f"- precision: {report['metrics']['precision']}",
        f"- recall: {report['metrics']['recall']}",
        f"- f1: {report['metrics']['f1']}",
        f"- localization_mean_m: {report['localization_error_m']['mean']}",
        f"- localization_median_m: {report['localization_error_m']['median']}",
        f"- inference_ms_mean: {report['performance']['inference_ms_mean']}",
        '',
        '## Per Class',
        '',
        '| class | TP | FP | FN | precision | recall | F1 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for cls, row in sorted(report['per_class'].items()):
        lines.append(
            f"| {cls} | {row['tp']} | {row['fp']} | {row['fn']} | "
            f"{row['precision']} | {row['recall']} | {row['f1']} |"
        )
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate trash detector JSONL log against Gazebo ground truth.')
    parser.add_argument(
        '--ground-truth',
        default='config/trash_ground_truth.yaml',
        help='YAML file with objects/id/class/position.',
    )
    parser.add_argument(
        '--detections',
        default='/tmp/trash_detections.jsonl',
        help='JSONL log produced by trash_detector.',
    )
    parser.add_argument(
        '--match-distance',
        type=float,
        default=None,
        help='Override YAML match_distance_m.',
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Output JSON report path. Default: data/eval/<timestamp>/report.json.',
    )
    parser.add_argument(
        '--markdown',
        default=None,
        help='Optional Markdown summary path.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ground_truth, default_distance = load_ground_truth(args.ground_truth)
    detections = load_detections(args.detections)
    match_distance = args.match_distance if args.match_distance is not None else default_distance
    report = evaluate(ground_truth, detections, match_distance)

    if args.output is None:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = Path('data') / 'eval' / stamp
        output = out_dir / 'report.json'
    else:
        output = Path(args.output)
        out_dir = output.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )

    markdown = Path(args.markdown) if args.markdown else output.with_suffix('.md')
    write_markdown(report, markdown)
    print(f'Wrote {output}')
    print(f'Wrote {markdown}')


if __name__ == '__main__':
    main()
