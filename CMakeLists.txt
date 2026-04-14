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
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
