import json
import math
import os
import sys
import io

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
    node._odom_x0 = None
    node._odom_y0 = None
    node._cam_tx = 0.60
    node._cam_tz = 0.83
    node._cam_pitch = 1.10
    node._merge = 1.2
    node._min_depth = 0.1
    node._max_depth = 10.0
    node._save_crops = False
    node._last_registered_id = None
    node._camera_mode = 'rgbd'
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


def test_world_to_pixel_projects_camera_axis_to_image_center():
    node = _make_detector_stub()
    depth = 2.0
    x_body = node._cam_tx + math.cos(node._cam_pitch) * depth
    z_body = node._cam_tz - math.sin(node._cam_pitch) * depth
    wx = node._robot_x + x_body
    wy = node._robot_y
    wz = detector.BODY_Z + z_body

    u, v, z_opt = detector.Detector._world_to_pixel(node, wx, wy, wz)

    assert u == pytest_approx(50.0)
    assert v == pytest_approx(40.0)
    assert z_opt == pytest_approx(depth)


def test_world_to_rviz_odom_applies_raw_odom_offset_for_markers():
    node = _make_detector_stub()
    node._odom_x0 = -0.5
    node._odom_y0 = 2.0

    mx, my = detector.Detector._world_to_rviz_odom(node, 4.0, -3.0)

    assert mx == pytest_approx(3.5)
    assert my == pytest_approx(-1.0)


def test_demo_ground_truth_assist_registers_visible_object_and_marks_source():
    node = _make_detector_stub()
    depth = 2.0
    x_body = node._cam_tx + math.cos(node._cam_pitch) * depth
    z_body = node._cam_tz - math.sin(node._cam_pitch) * depth
    node._demo_gt_objects = [
        {
            'id': 'demo_plastic_bottle_1',
            'class': 'plastic_bottle',
            'label': 'plastic_bottle',
            'x': node._robot_x + x_body,
            'y': node._robot_y,
            'z': detector.BODY_Z + z_body,
        }
    ]
    node._demo_assist_max_range = 10.0
    node._demo_assist_conf = 0.93
    node._log_fp = io.StringIO()
    frame = np.zeros((90, 120, 3), dtype=np.uint8)

    assert detector.Detector._apply_demo_ground_truth_assist(
        node, frame, frame_seq=3, frame_stamp=None, inference_ms=11.0)

    assert len(node._trash) == 1
    assert node._trash[0]['category'] == 'plastic_bottle'
    record = json.loads(node._log_fp.getvalue().strip())
    assert record['source'] == 'demo_ground_truth_assist'
    assert record['class'] == 'plastic_bottle'
    assert frame.sum() > 0


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


def test_detection_stat_lines_include_coordinates_and_runtime():
    node = _make_detector_stub()

    lines = detector.Detector._detection_stat_lines(
        node,
        object_id=7,
        label='Plastic bottle',
        category='plastic_bottle',
        conf=0.876,
        bbox_xyxy=(10.0, 20.0, 90.0, 140.0),
        depth=1.234,
        world=(2.5, -1.25),
        inference_ms=18.4,
        frame_seq=42,
    )
    text = '\n'.join(lines)

    assert 'object_id: 7' in text
    assert 'class: plastic_bottle' in text
    assert 'confidence: 0.876' in text
    assert 'depth: 1.23 m' in text
    assert 'world: x=2.50 y=-1.25 m' in text
    assert 'inference: 18.4 ms' in text


def test_detection_overlay_and_card_change_pixels():
    node = _make_detector_stub()
    frame = np.zeros((120, 180, 3), dtype=np.uint8)

    detector.Detector._draw_detection_info(
        node,
        frame,
        label='Can',
        category='aluminum_can',
        conf=0.75,
        bbox_xyxy=(20.0, 30.0, 80.0, 100.0),
        depth=0.9,
        world=(1.0, 2.0),
        object_id=3,
        inference_ms=12.0,
        frame_seq=5,
    )

    assert frame.sum() > 0

    crop = np.full((50, 70, 3), 120, dtype=np.uint8)
    lines = detector.Detector._detection_stat_lines(
        node,
        object_id=3,
        label='Can',
        category='aluminum_can',
        conf=0.75,
        bbox_xyxy=(20.0, 30.0, 80.0, 100.0),
        depth=0.9,
        world=(1.0, 2.0),
        inference_ms=12.0,
        frame_seq=5,
    )
    card = detector.Detector._make_detection_card(node, crop, lines)

    assert card.shape[0] >= crop.shape[0]
    assert card.shape[1] > crop.shape[1]


def test_save_detection_writes_annotated_photos_and_metadata(tmp_path):
    node = _make_detector_stub()
    node._save_dir = str(tmp_path)
    frame = np.full((120, 180, 3), 80, dtype=np.uint8)

    detector.Detector._save_detection(
        node,
        tid=2,
        label='Plastic bottle',
        category='plastic_bottle',
        conf=0.91,
        frame_bgr=frame,
        bbox=(50.0, 60.0, 20.0, 25.0, 100.0, 110.0),
        depth=1.5,
        world=(3.0, -2.0),
        inference_ms=17.5,
        frame_seq=11,
    )

    assert len(list(tmp_path.glob('*_full.jpg'))) == 1
    assert len(list(tmp_path.glob('*_crop.jpg'))) == 1
    assert len(list(tmp_path.glob('*_card.jpg'))) == 1
    meta_files = list(tmp_path.glob('*_meta.json'))
    assert len(meta_files) == 1

    meta = json.loads(meta_files[0].read_text(encoding='utf-8'))
    assert meta['object_id'] == 2
    assert meta['class'] == 'plastic_bottle'
    assert meta['world'] == {'x': 3.0, 'y': -2.0}
    assert meta['depth_m'] == 1.5


def pytest_approx(value):
    import pytest

    return pytest.approx(value, rel=1e-6, abs=1e-6)
