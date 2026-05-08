#!/usr/bin/env python3
"""Capture ROS image topics to files for demo evidence without manual screenshots."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class ImageCapture(Node):
    def __init__(self, topics: list[str], output_dir: Path, max_frames: int,
                 every_n: int):
        super().__init__('trash_image_capture')
        self._output_dir = output_dir
        self._max_frames = max(1, int(max_frames))
        self._every_n = max(1, int(every_n))
        self._seen = {topic: 0 for topic in topics}
        self._saved = {topic: 0 for topic in topics}
        output_dir.mkdir(parents=True, exist_ok=True)

        for topic in topics:
            safe_topic = self._safe_topic(topic)
            (output_dir / safe_topic).mkdir(parents=True, exist_ok=True)
            self.create_subscription(
                Image,
                topic,
                lambda msg, topic=topic: self._image_cb(topic, msg),
                10,
            )
        self.get_logger().info(
            f'Capturing {topics} to {output_dir}, max_frames={self._max_frames}'
        )

    @staticmethod
    def _safe_topic(topic: str) -> str:
        safe = topic.strip('/').replace('/', '__')
        return safe or 'image'

    @staticmethod
    def _stamp_text(msg: Image) -> str:
        stamp = msg.header.stamp
        return f'{int(stamp.sec):010d}_{int(stamp.nanosec):09d}'

    @staticmethod
    def _to_bgr(msg: Image) -> np.ndarray | None:
        if msg.encoding in ('rgb8', 'RGB8'):
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        if msg.encoding in ('bgr8', 'BGR8'):
            return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3).copy()
        if msg.encoding in ('mono8', '8UC1'):
            gray = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if msg.encoding == '32FC1':
            depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
            finite = depth[np.isfinite(depth)]
            if finite.size == 0:
                return None
            lo, hi = float(np.percentile(finite, 2)), float(np.percentile(finite, 98))
            if hi <= lo:
                hi = lo + 1.0
            norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
            gray = (norm * 255).astype(np.uint8)
            return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
        return None

    def _image_cb(self, topic: str, msg: Image) -> None:
        self._seen[topic] += 1
        if self._seen[topic] % self._every_n:
            return
        if self._saved[topic] >= self._max_frames:
            return

        img = self._to_bgr(msg)
        if img is None:
            self.get_logger().warn(f'Unsupported image encoding on {topic}: {msg.encoding}')
            return

        self._saved[topic] += 1
        safe_topic = self._safe_topic(topic)
        filename = f'{self._saved[topic]:03d}_{self._stamp_text(msg)}.jpg'
        path = self._output_dir / safe_topic / filename
        cv2.imwrite(str(path), img)
        self.get_logger().info(f'saved {path}')
        if all(count >= self._max_frames for count in self._saved.values()):
            rclpy.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description='Save ROS Image topics as jpg files.')
    parser.add_argument('--topic', action='append', default=[],
                        help='Image topic to capture. Can repeat.')
    parser.add_argument('--output-dir', default='/tmp/trash_demo_capture')
    parser.add_argument('--max-frames', type=int, default=20)
    parser.add_argument('--every-n', type=int, default=5)
    args = parser.parse_args()

    topics = args.topic or ['/rgbd/image/image', '/detections_img']
    rclpy.init()
    node = ImageCapture(topics, Path(args.output_dir), args.max_frames, args.every_n)
    try:
        rclpy.spin(node)
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
