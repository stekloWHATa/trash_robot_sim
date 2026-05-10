# Detection Demo Report Assets

This folder is generated from the detector JSONL log and Gazebo ground truth.

## Summary

- detections_raw: 4818
- detections_deduped: 2
- classes_detected: ['cardboard_box', 'cigarette_butt', 'paper_packaging']
- mean_confidence: 0.399
- median_confidence: 0.430
- mean_inference_ms: 12.134
- median_inference_ms: 11.862
- estimated_fps_from_log: 1.692

## Per Class

| class | detections | mean conf | median conf | mean depth, m |
| --- | ---: | ---: | ---: | ---: |
| cardboard_box | 1 | 0.357 | 0.357 | 1.531 |
| cigarette_butt | 1 | 0.384 | 0.384 | 1.199 |
| paper_packaging | 4816 | 0.399 | 0.430 | 1.105 |

## Generated Files

- `summary.json`
- `detections_by_class.csv`
- `detections_by_class.svg`
- `confidence_by_class.svg`
- `depth_by_class.svg`
- `world_scatter.svg`
- `latency_histogram.svg`
- `detections_timeline.svg`
- `evaluation_snapshot.json`
