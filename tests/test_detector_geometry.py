import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import detector


class _Logger:
    def info(self, *_args, **_kwargs):
        pass


def _make_detector_stub():
    node = object.__new__(detector.Detector)
    node._K = np.array(
        [[100.0, 0.0, 50.0],
         [0.0, 100.0, 40.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    node._robot_x = 1.0
    node._robot_y = -2.0
    node._robot_yaw = 0.0
    node._cam_tx = 0.60
    node._cam_tz = 0.83
    node._cam_pitch = 1.10
    node._merge = 1.2
    node._min_depth = 0.1
    node._max_depth = 10.0
    node._save_crops = False
    node._last_registered_id = None
    node._trash = {}
    node._trash_counter = 0
    node.get_logger = lambda: _Logger()
    return node


def test_sample_depth_returns_median_of_valid_values():
    node = _make_detector_stub()
    depth = np.array(
        [
            [0.0, 0.2, np.nan],
            [1.0, 2.0, 20.0],
            [3.0, np.inf, 4.0],
        ],
        dtype=np.float32,
    )

    assert detector.Detector._sample_depth(node, depth, 1, 1, patch=1) == 2.0


def test_sample_depth_rejects_too_few_valid_values():
    node = _make_detector_stub()
    depth = np.array([[0.0, np.nan], [20.0, 0.05]], dtype=np.float32)

    assert detector.Detector._sample_depth(node, depth, 0, 0, patch=1) is None


def test_pixel_to_world_depth_uses_camera_extrinsics():
    node = _make_detector_stub()

    wx, wy = detector.Detector._pixel_to_world(node, 50.0, 40.0, 2.0)

    expected_x = node._robot_x + node._cam_tx + math.cos(node._cam_pitch) * 2.0
    assert wx == pytest_approx(expected_x)
    assert wy == pytest_approx(node._robot_y)


def test_pixel_to_world_depth_respects_robot_yaw():
    node = _make_detector_stub()
    node._robot_yaw = math.pi / 2.0

    wx, wy = detector.Detector._pixel_to_world(node, 50.0, 40.0, 2.0)

    x_body = node._cam_tx + math.cos(node._cam_pitch) * 2.0
    assert wx == pytest_approx(node._robot_x)
    assert wy == pytest_approx(node._robot_y + x_body)


def test_pixel_to_world_ground_intersects_floor_plane():
    node = _make_detector_stub()
    node._cam_tx = 0.60
    node._cam_tz = 0.0
    node._cam_pitch = 0.45

    wx, wy = detector.Detector._pixel_to_world_ground(node, 50.0, 40.0)

    t = detector.BODY_Z / math.sin(node._cam_pitch)
    expected_x = node._robot_x + node._cam_tx + t * math.cos(node._cam_pitch)
    assert wx == pytest_approx(expected_x)
    assert wy == pytest_approx(node._robot_y)


def test_register_merges_nearby_detections_and_adds_far_ones():
    node = _make_detector_stub()

    assert detector.Detector._register(node, 1.0, 2.0, 'plastic_bottle', 'Plastic bottle', 0.5)
    assert not detector.Detector._register(node, 1.2, 2.0, 'plastic_bottle', 'Plastic bottle', 0.8)
    assert detector.Detector._register(node, 5.0, 2.0, 'aluminum_can', 'Can', 0.6)

    assert len(node._trash) == 2
    assert node._trash[0]['count'] == 2
    assert node._trash[0]['conf'] == 0.8
    assert node._last_registered_id == 1


def test_parse_class_conf_ignores_bad_items():
    parsed = detector.Detector._parse_class_conf(
        'cigarette butt:0.20, bad, plastic-bottle:0.45, can:nope'
    )

    assert parsed == {
        'cigarette_butt': 0.20,
        'plastic_bottle': 0.45,
    }


def pytest_approx(value):
    import pytest

    return pytest.approx(value, rel=1e-6, abs=1e-6)
