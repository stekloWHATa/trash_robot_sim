import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import audit_dataset


def _make_yolo_dataset(root: Path):
    for split in ('train', 'valid', 'test'):
        (root / split / 'images').mkdir(parents=True)
        (root / split / 'labels').mkdir(parents=True)
    with open(root / 'data.yaml', 'w') as f:
        yaml.safe_dump(
            {
                'path': str(root),
                'train': 'train/images',
                'val': 'valid/images',
                'test': 'test/images',
                'nc': 2,
                'names': ['bottle', 'can'],
            },
            f,
        )


def test_audit_yolo_dataset_counts_boxes_and_invalid_labels(tmp_path):
    _make_yolo_dataset(tmp_path)
    (tmp_path / 'train' / 'images' / 'ok.jpg').write_bytes(b'img')
    (tmp_path / 'train' / 'images' / 'missing_label.jpg').write_bytes(b'img')
    (tmp_path / 'train' / 'labels' / 'ok.txt').write_text(
        '0 0.5 0.5 0.2 0.2\n'
        '1 0.1 0.1 0.1 0.1\n'
        '3 0.5 0.5 0.2 0.2\n'
        'bad line\n'
    )
    (tmp_path / 'train' / 'labels' / 'orphan.txt').write_text('')

    audit = audit_dataset.audit_yolo_dataset(str(tmp_path / 'data.yaml'))

    assert audit['totals']['images'] == 2
    assert audit['totals']['labels'] == 2
    assert audit['totals']['boxes'] == 2
    assert audit['totals']['invalid_labels'] == 2
    assert audit['totals']['empty_labels'] == 1
    assert audit['totals']['orphan_images'] == 1
    assert audit['totals']['orphan_labels'] == 1
    assert audit['class_counts'] == {'bottle': 1, 'can': 1}


def test_write_reports_creates_json_and_markdown(tmp_path):
    _make_yolo_dataset(tmp_path)
    audit = audit_dataset.audit_yolo_dataset(str(tmp_path / 'data.yaml'))
    json_path = tmp_path / 'audit.json'
    md_path = tmp_path / 'audit.md'

    audit_dataset.write_reports(audit, json_path=str(json_path), md_path=str(md_path))

    assert json_path.is_file()
    assert md_path.is_file()
    assert 'YOLO Dataset Audit' in md_path.read_text()
