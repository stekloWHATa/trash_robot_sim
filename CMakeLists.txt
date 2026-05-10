cmake_minimum_required(VERSION 3.8)
project(trash_robot_sim)

find_package(ament_cmake REQUIRED)

# Устанавливаем ресурсы пакета
install(DIRECTORY
  urdf
  launch
  worlds
  models
  config
  rviz
  DESTINATION share/${PROJECT_NAME}
)

# Python-скрипты
install(PROGRAMS
  scripts/navigator.py
  scripts/map_builder.py
  scripts/map_check.py
  scripts/detector.py
  scripts/train_yolo.py
  scripts/audit_dataset.py
  scripts/check_dataset_readiness.py
  scripts/capture_ros_images.py
  scripts/run_demo_capture.py
  scripts/evaluate_detection_run.py
  scripts/generate_demo_report_assets.py
  scripts/generate_nir_graphs.py
  scripts/generate_textured_trash_assets.py
  scripts/scripted_motion.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
