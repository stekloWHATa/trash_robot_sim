import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import check_dataset_readiness


def _make_dataset(root: Path, names):
    for split in ('train', 'valid', 'test'):
        (root / split / 'images').mkdir(parents=True)
        (root / split / 'labels').mkdir(parents=True)
    (root / 'data.yaml').write_text(
        yaml.safe_dump({
            'path': str(root),
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'nc': len(names),
            'names': names,
        })
    )


def test_check_readiness_flags_missing_counts_and_invalid_labels(tmp_path):
    _make_dataset(tmp_path, ['cigarette_butt', 'plastic_bottle'])
    (tmp_path / 'train' / 'images' / 'a.jpg').write_bytes(b'img')
    (tmp_path / 'train' / 'labels' / 'a.txt').write_text(
        '0 0.5 0.5 0.1 0.1\n'
        '3 0.5 0.5 0.1 0.1\n'
    )

    report = check_dataset_readiness.check_readiness(
        tmp_path / 'data.yaml',
        min_boxes={'cigarette_butt': 2, 'plastic_bottle': 1},
    )

    assert not report['ready_for_yolov8s']
    assert any('invalid labels' in item for item in report['problems'])
    assert any('cigarette_butt' in item for item in report['problems'])
    assert any('plastic_bottle' in item for item in report['problems'])


def test_check_readiness_accepts_balanced_valid_dataset(tmp_path):
    _make_dataset(tmp_path, ['cigarette_butt', 'plastic_bottle'])
    for split in ('train', 'valid', 'test'):
        (tmp_path / split / 'images' / f'{split}.jpg').write_bytes(b'img')
        (tmp_path / split / 'labels' / f'{split}.txt').write_text(
            '0 0.5 0.5 0.1 0.1\n'
            '1 0.4 0.4 0.2 0.2\n'
        )

    report = check_dataset_readiness.check_readiness(
        tmp_path / 'data.yaml',
        min_boxes={'cigarette_butt': 3, 'plastic_bottle': 3},
    )

    assert report['ready_for_yolov8s']
    assert report['problems'] == []
