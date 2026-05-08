#!/usr/bin/env python3
"""Launch detection demo and automatically capture ROS image topics."""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run detection demo and save frames without manual screenshots.'
    )
    parser.add_argument('--output-dir', default='/tmp/trash_demo_capture')
    parser.add_argument('--max-frames', type=int, default=30)
    parser.add_argument('--every-n', type=int, default=5)
    parser.add_argument('--startup-wait', type=float, default=8.0)
    parser.add_argument('--scripted-motion', default='true')
    parser.add_argument('--topic', action='append', default=[],
                        help='Image topic to capture. Can repeat.')
    args = parser.parse_args()

    topics = args.topic or ['/rgbd/image/image', '/detections_img']
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    launch_cmd = [
        'ros2', 'launch', 'trash_robot_sim', 'detection_demo.launch.py',
        f'scripted_motion:={args.scripted_motion}',
    ]
    capture_cmd = [
        'ros2', 'run', 'trash_robot_sim', 'capture_ros_images.py',
        '--output-dir', str(output_dir),
        '--max-frames', str(args.max_frames),
        '--every-n', str(args.every_n),
    ]
    for topic in topics:
        capture_cmd.extend(['--topic', topic])

    print('[INFO] launching demo:', ' '.join(launch_cmd))
    launch_proc = subprocess.Popen(launch_cmd)
    try:
        print(f'[INFO] waiting {args.startup_wait:.1f}s for Gazebo/ROS topics...')
        time.sleep(args.startup_wait)
        print('[INFO] capturing images:', ' '.join(capture_cmd))
        capture_rc = subprocess.call(capture_cmd)
        if capture_rc != 0:
            raise SystemExit(capture_rc)
    finally:
        _terminate(launch_proc)

    print(f'[OK] captured demo frames to {output_dir}')


if __name__ == '__main__':
    main()
