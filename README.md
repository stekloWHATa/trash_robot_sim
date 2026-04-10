# trash_robot_sim

**Система локализации и идентификации бытового мусора в рабочей сцене мобильного робота**

Дипломный проект. Симуляция мобильного робота в Gazebo Harmonic (ROS 2), который автономно
объезжает заданную область, строит карту занятости и детектирует объекты бытового мусора
по цветовым признакам с использованием RGBD-камеры.

---

## Архитектура системы

```
Gazebo Harmonic
    │
    │  ros_gz_bridge
    ▼
/odom ──────────────────► astar_navigator ──► /cmd_vel ──► Gazebo
/scan ──────────────────► map_builder
/rgbd/image ────────────► map_builder
/rgbd/depth_image ──────► map_builder
/tf, /joint_states ─────► robot_state_publisher ──► /robot_description
                              │
                              ▼
                         RViz 2
                    /map, /trash_markers,
                    /camera/image, /rgbd/image
```

### Узлы ROS 2

| Узел | Исполняемый файл | Назначение |
|---|---|---|
| `astar_navigator` | `scripts/navigator.py` | A\*-планировщик + boustrophedon-покрытие |
| `map_builder` | `scripts/map_builder.py` | Log-odds карта занятости + детекция мусора |
| `robot_state_publisher` | системный | Публикует `/robot_description` и TF из URDF |
| `gz_bridge` | `ros_gz_bridge` | Мост Gazebo ↔ ROS 2 |

### Топики

| Топик | Тип | Описание |
|---|---|---|
| `/cmd_vel` | `geometry_msgs/Twist` | Команды скорости → робот |
| `/odom` | `nav_msgs/Odometry` | Одометрия из Gazebo |
| `/scan` | `sensor_msgs/LaserScan` | Данные лидара |
| `/camera/image` | `sensor_msgs/Image` | Навигационная камера (FOV 60°) |
| `/rgbd/image` | `sensor_msgs/Image` | RGBD-камера, цвет (FOV 86°) |
| `/rgbd/depth_image` | `sensor_msgs/Image` | RGBD-камера, глубина (32FC1, метры) |
| `/map` | `nav_msgs/OccupancyGrid` | Карта занятости (10 Гц) |
| `/trash_markers` | `visualization_msgs/MarkerArray` | Маркеры найденного мусора в RViz |
| `/trash_report` | `std_msgs/String` | Текстовый отчёт (каждые 5 с) |
| `/scan_area` | `geometry_msgs/Polygon` | Область сканирования (оператор → навигатор) |
| `/goal_pose` | `geometry_msgs/PoseStamped` | Разовая цель навигации (клик в RViz) |

---

## Зависимости

```bash
# ROS 2 Jazzy / Humble
sudo apt install \
  ros-$ROS_DISTRO-ros-gz \
  ros-$ROS_DISTRO-robot-state-publisher \
  ros-$ROS_DISTRO-joint-state-publisher \
  ros-$ROS_DISTRO-xacro \
  ros-$ROS_DISTRO-rviz2 \
  python3-numpy
```

---

## Сборка

```bash
cd ~/ros2_ws
colcon build --packages-select trash_robot_sim
source install/setup.bash
```

---

## Запуск

### Терминал 1 — симуляция + навигация

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch trash_robot_sim gazebo.launch.py
```

Запускает:
- Gazebo Harmonic с миром `trash_world.sdf`
- Робота в позиции (-0.5, -2.0)
- Узлы `astar_navigator`, `map_builder`, `robot_state_publisher`
- Мост Gazebo ↔ ROS 2

### Терминал 2 — визуализация

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch trash_robot_sim rviz.launch.py
```

Открывает RViz с настроенным конфигом:
- **Map** — строящаяся карта занятости
- **LaserScan** — точки лидара (оранжевые)
- **TrashMarkers** — сферы с подписями найденного мусора
- **RobotModel** — модель робота
- **NavCamera / RGBDCamera** — изображения с камер
- **OdomTrail** — след движения робота

### Терминал 3 — задать область сканирования

После запуска симуляции отправить полигон — навигатор построит зигзаг-маршрут и начнёт объезд:

```bash
# Сканировать всю арену (-9..+9 по обеим осям)
ros2 topic pub /scan_area geometry_msgs/msg/Polygon \
  "{points: [{x: -9.0, y: -9.0, z: 0.0}, {x: 9.0, y: 9.0, z: 0.0}]}" --once

# Сканировать правый верхний квадрант
ros2 topic pub /scan_area geometry_msgs/msg/Polygon \
  "{points: [{x: 0.0, y: 0.0, z: 0.0}, {x: 9.0, y: 9.0, z: 0.0}]}" --once
```

Либо кликнуть **"2D Goal Pose"** в RViz — робот поедет в указанную точку.

---

## Параметры

Все параметры вынесены в `config/params.yaml` и загружаются автоматически при запуске.

| Параметр | По умолч. | Описание |
|---|---|---|
| `row_spacing` | 2.5 м | Шаг между строками зигзага |
| `area_margin` | 0.8 м | Отступ от краёв области |
| `max_linear_velocity` | 0.65 м/с | Макс. линейная скорость |
| `max_angular_velocity` | 0.9 рад/с | Макс. угловая скорость |
| `goal_radius` | 0.90 м | Радиус достижения цели |
| `goal_timeout` | 120 с | Таймаут на одну точку пути |
| `detect_threshold` | 0.08 | Мин. доля пикселей для детекции |
| `merge_distance` | 2.5 м | Радиус объединения обнаружений |

---

## Классы мусора

| Класс | Цвет маркера | Описание |
|---|---|---|
| `plastic_bottle_green` | зелёный | Пластиковая бутылка (зелёная) |
| `plastic_bottle_blue` | синий | Пластиковая бутылка (синяя) |
| `can_red` | красный | Металлическая банка (красная) |
| `can_silver` | серый | Металлическая банка (серебро) |
| `cardboard_box` | коричневый | Картонная коробка |
| `plastic_bag` | светло-серый | Полиэтиленовый пакет |
| `paper` | бежевый | Бумага / газета |
| `bottle_glass` | тёмно-зелёный | Стеклянная бутылка |

---

## Сцена (trash_world.sdf)

- Арена **20 × 20 м** (от -10 до +10 по X и Y)
- **5 барьеров** внутри арены
- **10 объектов мусора**, равномерно рассредоточенных по арене
- Робот стартует в позиции **(0, -2)** по одометрии

---

## Структура пакета

```
trash_robot_sim/
├── CMakeLists.txt
├── package.xml
├── config/
│   └── params.yaml          # параметры узлов
├── launch/
│   ├── gazebo.launch.py     # запуск симуляции
│   └── rviz.launch.py       # запуск RViz
├── models/
│   └── robot/
│       └── only_robot.sdf   # SDF-модель робота (активная)
├── rviz/
│   └── robot.rviz           # конфигурация RViz
├── scripts/
│   ├── navigator.py         # A* + покрывающая навигация
│   └── map_builder.py       # карта занятости + детекция мусора
├── urdf/
│   └── trash_robot.urdf.xacro  # URDF для robot_state_publisher
└── worlds/
    └── trash_world.sdf      # мир Gazebo
```
