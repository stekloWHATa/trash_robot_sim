import json
import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import evaluate_detection_run


def test_evaluate_matches_by_class_and_distance_after_deduping_object_ids(tmp_path):
    gt_path = tmp_path / 'gt.yaml'
    gt_path.write_text(
        yaml.safe_dump(
            {
                'match_distance_m': 0.5,
                'objects': [
                    {'id': 'can_1', 'class': 'aluminum_can', 'position': {'x': 0, 'y': 0}},
                    {'id': 'bottle_1', 'class': 'plastic_bottle', 'position': {'x': 2, 'y': 0}},
                ],
            }
        ),
        encoding='utf-8',
    )
    log_path = tmp_path / 'detections.jsonl'
    records = [
        {
            'object_id': 5,
            'class': 'can_cup',
            'confidence': 0.60,
            'world': {'x': 0.1, 'y': 0.1},
            'timestamp_wall': 10.0,
            'inference_ms': 20.0,
        },
        {
            'object_id': 5,
            'class': 'aluminum_can',
            'confidence': 0.40,
            'world': {'x': 0.2, 'y': 0.2},
            'timestamp_wall': 10.2,
            'inference_ms': 24.0,
        },
        {
            'object_id': 6,
            'class': 'plastic_bottle',
            'confidence': 0.90,
            'world': {'x': 2.2, 'y': 0.0},
            'timestamp_wall': 10.4,
            'inference_ms': 22.0,
        },
        {
            'object_id': None,
            'class': 'plastic_bag',
            'confidence': 0.70,
            'world': {'x': 3.0, 'y': 0.0},
            'timestamp_wall': 10.6,
            'inference_ms': 21.0,
        },
    ]
    log_path.write_text(
        ''.join(json.dumps(row) + '\n' for row in records),
        encoding='utf-8',
    )

    gt, match_distance = evaluate_detection_run.load_ground_truth(gt_path)
    detections = evaluate_detection_run.load_detections(log_path)
    report = evaluate_detection_run.evaluate(gt, detections, match_distance)

    assert report['counts']['detections_raw'] == 4
    assert report['counts']['detections_evaluated'] == 3
    assert report['counts']['tp'] == 2
    assert report['counts']['fp'] == 1
    assert report['counts']['fn'] == 0
    assert report['metrics']['precision'] == pytest.approx(2 / 3)
    assert report['metrics']['recall'] == pytest.approx(1.0)
    assert report['metrics']['f1'] == pytest.approx(0.8)
    assert report['localization_error_m']['median'] == pytest.approx(0.170710678, rel=1e-6)


def test_write_markdown_contains_per_class_table(tmp_path):
    report = {
        'match_distance_m': 0.5,
        'counts': {'ground_truth': 1, 'detections_raw': 1, 'tp': 1, 'fp': 0, 'fn': 0},
        'metrics': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0},
        'localization_error_m': {'mean': 0.1, 'median': 0.1},
        'performance': {'inference_ms_mean': 20.0},
        'per_class': {
            'aluminum_can': {'tp': 1, 'fp': 0, 'fn': 0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
        },
    }
    md_path = tmp_path / 'report.md'

    evaluate_detection_run.write_markdown(report, md_path)

    text = md_path.read_text(encoding='utf-8')
    assert 'Detection Demo Evaluation' in text
    assert '| aluminum_can | 1 | 0 | 0 | 1.0 | 1.0 | 1.0 |' in text
