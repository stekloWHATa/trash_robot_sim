from pathlib import Path
import xml.etree.ElementTree as ET

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_detection_demo_world_matches_ground_truth_and_local_models():
    world = ET.parse(ROOT / 'worlds' / 'detection_demo_world.sdf')
    root = world.getroot()
    includes = {}
    for include in world.findall('.//include'):
        name = include.findtext('name')
        uri = include.findtext('uri')
        if name and uri:
            includes[name] = uri

    gt = yaml.safe_load((ROOT / 'config' / 'trash_ground_truth.yaml').read_text(encoding='utf-8'))
    gt_ids = {obj['id'] for obj in gt['objects']}
    assert set(includes) == gt_ids

    model_names = {model.attrib.get('name') for model in root.findall('.//model')}
    assert 'back_wall' not in model_names
    assert 'left_wall' not in model_names

    ground_size = root.findtext(".//model[@name='ground_plane']//visual//plane/size")
    assert ground_size == '24 14'

    gt_classes = [obj['class'] for obj in gt['objects']]
    assert gt_classes == [
        'cigarette_butt',
        'aluminum_can',
        'plastic_bottle',
        'plastic_bag',
        'cardboard_box',
        'paper_packaging',
    ]

    for obj in gt['objects']:
        assert obj['id'] in includes
        model_name = includes[obj['id']].replace('model://', '')
        model_dir = ROOT / 'models' / 'trash' / model_name
        assert model_dir.is_dir()
        model_sdf = model_dir / 'model.sdf'
        assert model_sdf.is_file()
        assert (model_dir / 'model.config').is_file()

        model_tree = ET.parse(model_sdf)
        mesh_uri = model_tree.findtext('.//mesh/uri')
        assert mesh_uri is not None
        assert mesh_uri.endswith(('.glb', '.obj', '.dae'))
        mesh_path = model_dir / mesh_uri.replace(f'model://{model_name}/', '')
        assert mesh_path.is_file()

        scale_text = model_tree.findtext('.//mesh/scale')
        if scale_text is not None:
            scale = [float(value) for value in scale_text.split()]
            assert len(scale) == 3
            assert max(scale) <= 1.5
        if mesh_uri.endswith('.obj'):
            assert (mesh_path.parent / f'{model_name}.mtl').is_file()


def test_detection_demo_launch_is_detector_focused():
    launch_text = (ROOT / 'launch' / 'detection_demo.launch.py').read_text(encoding='utf-8')

    assert 'detector.py' in launch_text
    assert 'map_builder.py' in launch_text
    assert 'scripted_motion.py' in launch_text
    assert 'navigator.py' not in launch_text
    assert 'astar_navigator' not in launch_text
