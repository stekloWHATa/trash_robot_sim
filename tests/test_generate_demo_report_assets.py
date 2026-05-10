import json
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import generate_demo_report_assets


def test_generate_demo_report_assets_writes_plots_and_tables(tmp_path):
    gt_path = tmp_path / 'gt.yaml'
    gt_path.write_text(
        yaml.safe_dump(
            {
                'match_distance_m': 0.75,
                'objects': [
                    {'id': 'can_1', 'class': 'aluminum_can', 'position': {'x': 1.0, 'y': -1.0}},
                    {'id': 'bag_1', 'class': 'plastic_bag', 'position': {'x': 2.0, 'y': -2.0}},
                ],
            }
        ),
        encoding='utf-8',
    )
    detections_path = tmp_path / 'detections.jsonl'
    rows = [
        {
            'object_id': 1,
            'class': 'aluminum_can',
            'confidence': 0.82,
            'depth_m': 1.4,
            'world': {'x': 1.05, 'y': -1.02},
            'timestamp_wall': 100.0,
            'frame_seq': 1,
            'inference_ms': 12.5,
        },
        {
            'object_id': 2,
            'class': 'plastic_bag',
            'confidence': 0.71,
            'depth_m': 1.8,
            'world': {'x': 2.15, 'y': -2.05},
            'timestamp_wall': 100.3,
            'frame_seq': 2,
            'inference_ms': 15.0,
        },
    ]
    detections_path.write_text(
        ''.join(json.dumps(row) + '\n' for row in rows),
        encoding='utf-8',
    )
    out_dir = tmp_path / 'report'

    summary = generate_demo_report_assets.generate_report_assets(
        detections_path=detections_path,
        ground_truth_path=gt_path,
        output_dir=out_dir,
    )

    assert summary['detections_raw'] == 2
    assert summary['evaluation_metrics']['precision'] == 1.0
    for name in [
        'summary.json',
        'summary.md',
        'detections_by_class.csv',
        'detections_by_class.svg',
        'confidence_by_class.svg',
        'depth_by_class.svg',
        'world_scatter.svg',
        'latency_histogram.svg',
        'detections_timeline.svg',
        'evaluation_snapshot.json',
    ]:
        assert (out_dir / name).is_file()

    csv_text = (out_dir / 'detections_by_class.csv').read_text(encoding='utf-8')
    assert 'aluminum_can' in csv_text
    assert 'plastic_bag' in csv_text
