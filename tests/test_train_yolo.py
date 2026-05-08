import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import train_yolo


def _write_class_config(path: Path):
    path.write_text(
        yaml.safe_dump(
            {
                'canonical_classes': ['plastic_bottle', 'aluminum_can'],
                'aliases': {
                    'plastic_bottle': ['plastic bottle', 'pet bottle'],
                    'aluminum_can': ['can', 'aluminum can'],
                },
                'ignored_classes': ['background'],
            }
        )
    )


def _write_dataset(root: Path, names, labels_by_split):
    root.mkdir(parents=True)
    with open(root / 'data.yaml', 'w') as f:
        yaml.safe_dump(
            {
                'path': '../old-relative-path',
                'train': '../train/images',
                'val': '../valid/images',
                'test': '../test/images',
                'nc': len(names),
                'names': names,
            },
            f,
        )

    for split, labels in labels_by_split.items():
        img_dir = root / split / 'images'
        lbl_dir = root / split / 'labels'
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        for stem, label_text in labels.items():
            (img_dir / f'{stem}.jpg').write_bytes(b'fake image bytes')
            if label_text is not None:
                (lbl_dir / f'{stem}.txt').write_text(label_text)


def test_fix_yaml_paths_normalizes_roboflow_layout(tmp_path):
    ds = tmp_path / 'ds'
    _write_dataset(ds, ['Bottle'], {'train': {}})

    cfg = train_yolo.fix_yaml_paths(str(ds))

    assert cfg['path'] == str(ds)
    assert cfg['train'] == 'train/images'
    assert cfg['val'] == 'valid/images'
    assert cfg['test'] == 'test/images'

    with open(ds / 'data.yaml') as f:
        saved = yaml.safe_load(f)
    assert saved == cfg


def test_merge_datasets_dedupes_classes_and_remaps_labels(tmp_path, monkeypatch):
    ds_a = tmp_path / 'dataset_a'
    ds_b = tmp_path / 'dataset_b'
    merged = tmp_path / 'merged'
    monkeypatch.setattr(train_yolo, 'MERGED_DIR', str(merged))

    _write_dataset(
        ds_a,
        ['Bottle', 'Can'],
        {
            'train': {'a': '0 0.5 0.5 0.2 0.2\n1 0.1 0.2 0.3 0.4\n'},
            'valid': {},
            'test': {},
        },
    )
    _write_dataset(
        ds_b,
        ['can', 'Box'],
        {
            'train': {'b': '0 0.3 0.3 0.1 0.1\n1 0.4 0.4 0.2 0.2\n'},
            'valid': {},
            'test': {},
        },
    )

    merged_yaml = train_yolo.merge_datasets([str(ds_a), str(ds_b)])

    with open(merged_yaml) as f:
        cfg = yaml.safe_load(f)
    assert cfg['names'] == ['Bottle', 'Can', 'Box']
    assert cfg['nc'] == 3

    a_label = merged / 'train' / 'labels' / 'dataset_a__a.txt'
    b_label = merged / 'train' / 'labels' / 'dataset_b__b.txt'
    assert a_label.read_text().splitlines() == [
        '0 0.5 0.5 0.2 0.2',
        '1 0.1 0.2 0.3 0.4',
    ]
    assert b_label.read_text().splitlines() == [
        '1 0.3 0.3 0.1 0.1',
        '2 0.4 0.4 0.2 0.2',
    ]


def test_merge_datasets_supports_dict_names_from_yolo_yaml(tmp_path, monkeypatch):
    ds = tmp_path / 'dict_names_ds'
    merged = tmp_path / 'merged'
    monkeypatch.setattr(train_yolo, 'MERGED_DIR', str(merged))

    _write_dataset(ds, {1: 'Can', 0: 'Bottle'}, {'train': {'sample': '1 0.1 0.2 0.3 0.4\n'}})

    merged_yaml = train_yolo.merge_datasets([str(ds)])

    with open(merged_yaml) as f:
        cfg = yaml.safe_load(f)
    assert cfg['names'] == ['Bottle', 'Can']
    assert (merged / 'train' / 'labels' / 'dict_names_ds__sample.txt').read_text() == (
        '1 0.1 0.2 0.3 0.4\n'
    )


def test_merge_datasets_creates_empty_label_for_unannotated_image(tmp_path, monkeypatch):
    ds = tmp_path / 'unannotated_ds'
    merged = tmp_path / 'merged'
    monkeypatch.setattr(train_yolo, 'MERGED_DIR', str(merged))

    _write_dataset(ds, ['Bottle'], {'train': {'empty': None}})

    train_yolo.merge_datasets([str(ds)])

    assert (merged / 'train' / 'images' / 'unannotated_ds__empty.jpg').is_file()
    assert (merged / 'train' / 'labels' / 'unannotated_ds__empty.txt').read_text() == ''


def test_merge_datasets_converts_segmentation_polygon_to_bbox(tmp_path, monkeypatch):
    ds = tmp_path / 'seg_ds'
    merged = tmp_path / 'merged'
    monkeypatch.setattr(train_yolo, 'MERGED_DIR', str(merged))

    _write_dataset(
        ds,
        ['Bottle'],
        {
            'train': {'poly': '0 0.1 0.2 0.3 0.2 0.3 0.6 0.1 0.6\n'},
            'valid': {},
            'test': {},
        },
    )

    train_yolo.merge_datasets([str(ds)])

    assert (merged / 'train' / 'labels' / 'seg_ds__poly.txt').read_text() == (
        '0 0.200000 0.400000 0.200000 0.400000\n'
    )


def test_load_class_config_maps_aliases(tmp_path):
    cfg_path = tmp_path / 'classes.yaml'
    _write_class_config(cfg_path)

    cfg = train_yolo.load_class_config(str(cfg_path))

    assert cfg['classes'] == ['plastic_bottle', 'aluminum_can']
    assert train_yolo._map_class('PET Bottle', cfg) == 'plastic_bottle'
    assert train_yolo._map_class('Aluminum-can', cfg) == 'aluminum_can'
    assert train_yolo._map_class('unknown class', cfg) is None


def test_real_trash_class_config_covers_selected_dataset_aliases():
    root = Path(__file__).resolve().parents[1]
    cfg = train_yolo.load_class_config(str(root / 'config' / 'trash_classes.yaml'))

    expected = {
        'Cigarette_Butt': 'cigarette_butt',
        'PET_Bottle': 'plastic_bottle',
        'Glass_bottle': 'glass_bottle',
        'Metal_drinkcan': 'aluminum_can',
        'Drink can': 'aluminum_can',
        'Food Can': 'aluminum_can',
        'Plastic_Bag': 'plastic_bag',
        'Garbage bag': 'plastic_bag',
        'Other plastic wrapper': 'plastic_bag',
        'cardboard_paper': 'cardboard_box',
        'Corrugated carton': 'cardboard_box',
        'Paper_carton': 'paper_packaging',
        'Paper_cigarette_package': 'paper_packaging',
        'Normal paper': 'paper_packaging',
        'Tissues': 'paper_packaging',
        'Pop tab': 'other_trash',
    }

    for source_class, canonical in expected.items():
        assert train_yolo._map_class(source_class, cfg) == canonical


def test_mvp_class_config_keeps_six_high_confidence_classes():
    root = Path(__file__).resolve().parents[1]
    cfg = train_yolo.load_class_config(str(root / 'config' / 'trash_classes_mvp.yaml'))

    assert cfg['classes'] == [
        'cigarette_butt',
        'plastic_bottle',
        'aluminum_can',
        'plastic_bag',
        'cardboard_box',
        'paper_packaging',
    ]
    assert train_yolo._map_class('Glass_bottle', cfg) is None
    assert train_yolo._map_class('PET_Bottle', cfg) == 'plastic_bottle'
    assert train_yolo._map_class('Metal_drinkcan', cfg) == 'aluminum_can'


def test_resolve_base_model_falls_back_to_ultralytics_name_for_known_models(tmp_path):
    missing = tmp_path / 'models' / 'yolov8s.pt'

    assert train_yolo._resolve_base_model(str(missing)) == 'yolov8s.pt'


def test_merge_datasets_applies_class_mapping_and_skips_unmapped(tmp_path, monkeypatch):
    cfg_path = tmp_path / 'classes.yaml'
    _write_class_config(cfg_path)
    class_config = train_yolo.load_class_config(str(cfg_path))
    ds = tmp_path / 'mapped_ds'
    merged = tmp_path / 'merged'
    monkeypatch.setattr(train_yolo, 'MERGED_DIR', str(merged))

    _write_dataset(
        ds,
        ['PET Bottle', 'Can', 'Background'],
        {
            'train': {
                'sample': (
                    '0 0.5 0.5 0.2 0.2\n'
                    '1 0.2 0.2 0.1 0.1\n'
                    '2 0.7 0.7 0.1 0.1\n'
                )
            },
            'valid': {},
            'test': {},
        },
    )

    merged_yaml = train_yolo.merge_datasets([str(ds)], class_config=class_config)

    with open(merged_yaml) as f:
        cfg = yaml.safe_load(f)
    assert cfg['names'] == ['plastic_bottle', 'aluminum_can']
    label = merged / 'train' / 'labels' / 'mapped_ds__sample.txt'
    assert label.read_text().splitlines() == [
        '0 0.5 0.5 0.2 0.2',
        '1 0.2 0.2 0.1 0.1',
    ]

    with open(merged / 'merge_report.yaml') as f:
        report = yaml.safe_load(f)
    assert report['boxes_kept'] == 2
    assert report['boxes_skipped'] == 1
    assert report['skipped_by_class']['Background'] == 1


def test_convert_coco_to_yolo_supports_taco_style_annotations(tmp_path):
    cfg_path = tmp_path / 'classes.yaml'
    _write_class_config(cfg_path)
    class_config = train_yolo.load_class_config(str(cfg_path))
    images = tmp_path / 'images'
    out = tmp_path / 'taco_yolo'
    images.mkdir()
    (images / 'img1.jpg').write_bytes(b'fake image bytes')
    coco = tmp_path / 'annotations.json'
    coco.write_text(
        json_dump(
            {
                'images': [{'id': 1, 'file_name': 'img1.jpg', 'width': 100, 'height': 50}],
                'categories': [
                    {'id': 10, 'name': 'Plastic Bottle'},
                    {'id': 20, 'name': 'Other'},
                ],
                'annotations': [
                    {'id': 1, 'image_id': 1, 'category_id': 10, 'bbox': [10, 5, 20, 10]},
                    {'id': 2, 'image_id': 1, 'category_id': 20, 'bbox': [0, 0, 5, 5]},
                ],
            }
        )
    )

    data_yaml = train_yolo.convert_coco_to_yolo(
        str(coco), str(images), str(out), split='train', class_config=class_config
    )

    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    assert cfg['names'] == ['plastic_bottle', 'aluminum_can']
    assert (out / 'train' / 'images' / 'img1.jpg').is_file()
    assert (out / 'train' / 'labels' / 'img1.txt').read_text().splitlines() == [
        '0 0.200000 0.200000 0.200000 0.200000'
    ]


def test_convert_coco_to_yolo_auto_split_and_nested_taco_names(tmp_path):
    cfg_path = tmp_path / 'classes.yaml'
    _write_class_config(cfg_path)
    class_config = train_yolo.load_class_config(str(cfg_path))
    images = tmp_path / 'images'
    out = tmp_path / 'taco_yolo'
    images.mkdir()
    coco_images = []
    annotations = []
    for idx in range(10):
        batch = images / f'batch_{idx % 2}'
        batch.mkdir(exist_ok=True)
        (batch / f'img{idx}.jpg').write_bytes(b'fake image bytes')
        coco_images.append({
            'id': idx,
            'file_name': f'batch_{idx % 2}/img{idx}.jpg',
            'width': 100,
            'height': 100,
        })
        annotations.append({
            'id': idx,
            'image_id': idx,
            'category_id': 10,
            'bbox': [10, 10, 20, 20],
        })
    coco = tmp_path / 'annotations.json'
    coco.write_text(json_dump({
        'images': coco_images,
        'categories': [{'id': 10, 'name': 'PET Bottle'}],
        'annotations': annotations,
    }))

    train_yolo.convert_coco_to_yolo(
        str(coco), str(images), str(out), split='auto', class_config=class_config
    )

    assert len(list((out / 'train' / 'images').glob('*.jpg'))) == 8
    assert len(list((out / 'valid' / 'images').glob('*.jpg'))) == 1
    assert len(list((out / 'test' / 'images').glob('*.jpg'))) == 1
    assert (out / 'train' / 'labels' / 'batch_0__img0.txt').is_file()
    with open(out / 'auto_coco_convert_report.yaml') as f:
        report = yaml.safe_load(f)
    assert report['images'] == {'train': 8, 'valid': 1, 'test': 1}


def test_main_only_extra_datasets_skips_default_dirs(tmp_path, monkeypatch):
    extra = tmp_path / 'extra_ds'
    _write_dataset(extra, ['Bottle'], {'train': {'sample': '0 0.5 0.5 0.2 0.2\n'}})
    default_dir = tmp_path / 'default_ds'
    _write_dataset(default_dir, ['Can'], {'train': {'default': '0 0.5 0.5 0.2 0.2\n'}})
    captured = {}

    def fake_merge(dataset_dirs, class_config=None, output_dir=None):
        captured['dataset_dirs'] = dataset_dirs
        captured['output_dir'] = output_dir
        return str(tmp_path / 'merged' / 'data.yaml')

    monkeypatch.setattr(train_yolo, 'DATA_ROOT', str(tmp_path))
    monkeypatch.setattr(train_yolo, 'MODELS_DIR', str(tmp_path / 'models'))
    monkeypatch.setattr(train_yolo, 'DATASETS', [('workspace', 'project', 1, 'default_ds')])
    monkeypatch.setattr(train_yolo, 'merge_datasets', fake_merge)
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'train_yolo.py',
            '--skip-download',
            '--prepare-only',
            '--only-extra-datasets',
            '--class-config',
            str(tmp_path / 'missing.yaml'),
            '--output-dir',
            str(tmp_path / 'merged'),
            '--extra-dataset',
            str(extra),
        ],
    )

    train_yolo.main()

    assert captured['dataset_dirs'] == [str(extra)]
    assert captured['output_dir'] == str(tmp_path / 'merged')


def json_dump(data):
    import json

    return json.dumps(data)
