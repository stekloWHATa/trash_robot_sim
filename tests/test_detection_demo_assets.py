from pathlib import Path
import xml.etree.ElementTree as ET

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_detection_demo_world_matches_ground_truth_and_local_models():
    world = ET.parse(ROOT / 'worlds' / 'detection_demo_world.sdf')
    includes = {}
    for include in world.findall('.//include'):
        name = include.findtext('name')
        uri = include.findtext('uri')
        if name and uri:
            includes[name] = uri

    gt = yaml.safe_load((ROOT / 'config' / 'trash_ground_truth.yaml').read_text(encoding='utf-8'))
    for obj in gt['objects']:
        assert obj['id'] in includes
        model_name = includes[obj['id']].replace('model://', '')
        model_dir = ROOT / 'models' / 'trash' / model_name
        assert model_dir.is_dir()
        assert (model_dir / 'model.sdf').is_file()
        assert (model_dir / 'model.config').is_file()


def test_detection_demo_launch_is_detector_focused():
    launch_text = (ROOT / 'launch' / 'detection_demo.launch.py').read_text(encoding='utf-8')

    assert 'detector.py' in launch_text
    assert 'map_builder.py' in launch_text
    assert 'scripted_motion.py' in launch_text
    assert 'navigator.py' not in launch_text
    assert 'astar_navigator' not in launch_text
