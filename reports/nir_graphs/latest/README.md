# Графики для НИР

Графики подготовлены под бланк НИР: YOLOv8, точность, быстродействие, пространственная локализация, сравнение с базовой моделью.

## Ключевые метрики

- лучший YOLOv8s mAP50: 0.827 (82.7%)
- лучшая точность YOLOv8s: 0.896 (89.6%)
- лучшая полнота YOLOv8s: 0.824 (82.4%)
- успешность локализации в демо: 1.000 (100.0%)
- распознаваемых классов: 6

## Важное примечание

`localization_demo_assist_acceptance.json` и графики локализации используют подготовленный demo-assist режим Gazebo. Они подходят для приемки видеодемо, а чистые метрики YOLO берутся только из `results.csv`.

## SVG-графики

- `01_requirements_compliance_percent.svg`
- `02_requirements_raw_values.svg`
- `03_yolov8s_map50_over_epochs.svg`
- `04_yolov8s_precision_recall_over_epochs.svg`
- `05_yolov8s_map50_95_over_epochs.svg`
- `06_yolo_train_losses.svg`
- `07_yolo_validation_losses.svg`
- `08_final_vs_best_detection_metrics.svg`
- `09_baseline_vs_yolov8s_map50.svg`
- `10_baseline_vs_yolov8s_precision_recall.svg`
- `11_dataset_split_images.svg`
- `12_dataset_split_bboxes.svg`
- `13_dataset_class_distribution.svg`
- `14_bbox_area_percentiles.svg`
- `15_demo_world_object_layout.svg`
- `16_localization_success_by_class.svg`
- `17_localization_error_per_object.svg`
- `18_localization_error_histogram.svg`
- `19_localization_error_cdf.svg`
- `20_object_level_tp_fp_fn.svg`
- `21_detection_latency_summary.svg`
- `22_inference_fps_estimate.svg`
- `23_training_time_per_epoch.svg`
- `24_learning_rate_schedule.svg`
- `25_metric_margin_to_target.svg`
- `26_nir_work_schedule.svg`
