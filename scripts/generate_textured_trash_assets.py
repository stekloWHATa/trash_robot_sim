#!/usr/bin/env python3
"""Generate local textured mesh assets for the Gazebo detection demo.

This is a deterministic fallback for machines without Blender installed.
The output is real OBJ/MTL + PNG texture, not SDF primitive blocks.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
TRASH_ROOT = ROOT / 'models' / 'trash'


def ensure_dirs(model: str) -> tuple[Path, Path]:
    base = TRASH_ROOT / model
    mesh_dir = base / 'meshes'
    texture_dir = base / 'textures'
    mesh_dir.mkdir(parents=True, exist_ok=True)
    texture_dir.mkdir(parents=True, exist_ok=True)
    return mesh_dir, texture_dir


def write_wavy_plane_obj(path: Path, mtl_name: str, width: float, depth: float,
                         nx: int = 14, ny: int = 10, height: float = 0.035) -> None:
    lines = [f'mtllib {mtl_name}', 'usemtl textured_surface']
    for iy in range(ny + 1):
        v = iy / ny
        y = (v - 0.5) * depth
        for ix in range(nx + 1):
            u = ix / nx
            x = (u - 0.5) * width
            edge_falloff = math.sin(math.pi * u) * math.sin(math.pi * v)
            wave = (
                0.55 * math.sin(5.0 * math.pi * u + 0.7)
                + 0.35 * math.cos(4.0 * math.pi * v + 1.1)
                + 0.25 * math.sin(3.0 * math.pi * (u + v))
            )
            z = 0.012 + height * edge_falloff * (0.55 + 0.45 * wave)
            lines.append(f'v {x:.6f} {y:.6f} {z:.6f}')
    for iy in range(ny + 1):
        v = iy / ny
        for ix in range(nx + 1):
            u = ix / nx
            lines.append(f'vt {u:.6f} {1.0 - v:.6f}')
    stride = nx + 1
    for iy in range(ny):
        for ix in range(nx):
            a = iy * stride + ix + 1
            b = a + 1
            c = a + stride
            d = c + 1
            lines.append(f'f {a}/{a} {b}/{b} {d}/{d}')
            lines.append(f'f {a}/{a} {d}/{d} {c}/{c}')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_flat_box_obj(path: Path, mtl_name: str, width: float, depth: float,
                       height: float) -> None:
    x = width / 2
    y = depth / 2
    z = height
    vertices = [
        (-x, -y, 0), (x, -y, 0), (x, y, 0), (-x, y, 0),
        (-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z),
    ]
    uv = [(0, 1), (1, 1), (1, 0), (0, 0)]
    faces = [
        (5, 6, 7, 8),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 4, 8, 7),
        (4, 1, 5, 8),
    ]
    lines = [f'mtllib {mtl_name}', 'usemtl textured_surface']
    lines.extend(f'v {vx:.6f} {vy:.6f} {vz:.6f}' for vx, vy, vz in vertices)
    lines.extend(f'vt {tu:.6f} {tv:.6f}' for tu, tv in uv)
    for face in faces:
        lines.append('f ' + ' '.join(f'{idx}/{(i % 4) + 1}' for i, idx in enumerate(face)))
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _rect_mesh(cx: float, cy: float, width: float, depth: float,
               z: float) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    x0, x1 = cx - width / 2, cx + width / 2
    y0, y1 = cy - depth / 2, cy + depth / 2
    return (
        [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)],
        [(0, 1, 2), (0, 2, 3)],
    )


def _ellipse_mesh(cx: float, cy: float, rx: float, ry: float, z: float,
                  segments: int = 48) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    vertices = [(cx, cy, z)]
    for idx in range(segments):
        angle = 2.0 * math.pi * idx / segments
        vertices.append((cx + rx * math.cos(angle), cy + ry * math.sin(angle), z))
    faces = []
    for idx in range(segments):
        faces.append((0, idx + 1, 1 + ((idx + 1) % segments)))
    return vertices, faces


def _wavy_surface_mesh(width: float, depth: float, nx: int = 20, ny: int = 12,
                       height: float = 0.055) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    vertices = []
    for iy in range(ny + 1):
        v = iy / ny
        y = (v - 0.5) * depth
        for ix in range(nx + 1):
            u = ix / nx
            x = (u - 0.5) * width
            edge_falloff = math.sin(math.pi * u) * math.sin(math.pi * v)
            wave = (
                0.55 * math.sin(5.0 * math.pi * u + 0.7)
                + 0.35 * math.cos(4.0 * math.pi * v + 1.1)
                + 0.25 * math.sin(3.0 * math.pi * (u + v))
            )
            z = 0.012 + height * edge_falloff * (0.55 + 0.45 * wave)
            vertices.append((x, y, z))
    faces = []
    stride = nx + 1
    for iy in range(ny):
        for ix in range(nx):
            a = iy * stride + ix
            b = a + 1
            c = a + stride
            d = c + 1
            faces.append((a, b, d))
            faces.append((a, d, c))
    return vertices, faces


def write_chip_bag_dae(path: Path) -> None:
    materials = {
        'red': (0.78, 0.06, 0.04, 1.0),
        'red_dark': (0.42, 0.02, 0.015, 1.0),
        'orange': (0.94, 0.30, 0.10, 1.0),
        'yellow': (1.00, 0.80, 0.10, 1.0),
        'gold': (0.98, 0.62, 0.12, 1.0),
        'brown': (0.32, 0.08, 0.025, 1.0),
        'white': (0.95, 0.93, 0.84, 1.0),
    }
    geometries: list[tuple[str, str, list[tuple[float, float, float]], list[tuple[int, int, int]]]] = []

    base_v, base_f = _wavy_surface_mesh(0.62, 0.38)
    geometries.append(('base', 'red', base_v, base_f))

    for idx, x in enumerate([-0.25, -0.13, -0.01, 0.11, 0.23]):
        v, f = _rect_mesh(x, 0.0, 0.050, 0.36, 0.071)
        geometries.append((f'stripe_{idx}', 'orange' if idx % 2 == 0 else 'red_dark', v, f))

    label_v, label_f = _ellipse_mesh(0.0, 0.0, 0.22, 0.105, 0.082)
    geometries.append(('label', 'yellow', label_v, label_f))
    inner_v, inner_f = _ellipse_mesh(0.0, 0.0, 0.17, 0.074, 0.084)
    geometries.append(('label_inner', 'gold', inner_v, inner_f))

    # Block letters: CHIPS, drawn as small raised rectangles on the mesh.
    letters = [
        # C
        (-0.145, 0.018, 0.011, 0.070), (-0.113, 0.048, 0.055, 0.012), (-0.113, -0.012, 0.055, 0.012),
        # H
        (-0.058, 0.018, 0.010, 0.070), (-0.020, 0.018, 0.010, 0.070), (-0.039, 0.018, 0.038, 0.012),
        # I
        (0.018, 0.018, 0.012, 0.070),
        # P
        (0.053, 0.018, 0.010, 0.070), (0.077, 0.049, 0.046, 0.012), (0.077, 0.018, 0.046, 0.012), (0.101, 0.035, 0.010, 0.032),
        # S
        (0.151, 0.049, 0.054, 0.012), (0.151, 0.018, 0.054, 0.012), (0.151, -0.013, 0.054, 0.012),
        (0.129, 0.035, 0.010, 0.030), (0.173, 0.000, 0.010, 0.030),
    ]
    for idx, (cx, cy, sx, sy) in enumerate(letters):
        v, f = _rect_mesh(cx, cy, sx, sy, 0.092)
        geometries.append((f'letter_{idx}', 'brown', v, f))

    for idx, (cx, cy) in enumerate([(-0.23, -0.10), (0.22, -0.08), (0.25, 0.09)]):
        v, f = _ellipse_mesh(cx, cy, 0.045, 0.025, 0.086, segments=28)
        geometries.append((f'chip_{idx}', 'gold', v, f))

    barcode_v, barcode_f = _rect_mesh(0.22, -0.135, 0.13, 0.045, 0.090)
    geometries.append(('barcode_panel', 'white', barcode_v, barcode_f))
    for idx in range(5):
        v, f = _rect_mesh(0.18 + idx * 0.020, -0.135, 0.004, 0.035, 0.096)
        geometries.append((f'barcode_{idx}', 'brown', v, f))

    def floats(values: list[tuple[float, float, float]]) -> str:
        return ' '.join(f'{coord:.6f}' for vertex in values for coord in vertex)

    effects = []
    material_nodes = []
    for name, rgba in materials.items():
        r, g, b, a = rgba
        effects.append(
            f'''    <effect id="{name}-effect">
      <profile_COMMON>
        <technique sid="common">
          <phong>
            <diffuse><color>{r:.3f} {g:.3f} {b:.3f} {a:.3f}</color></diffuse>
            <specular><color>0.080 0.080 0.080 1.000</color></specular>
            <shininess><float>18.0</float></shininess>
          </phong>
        </technique>
      </profile_COMMON>
    </effect>'''
        )
        material_nodes.append(f'    <material id="{name}-material" name="{name}"><instance_effect url="#{name}-effect"/></material>')

    geometry_nodes = []
    scene_nodes = []
    for geom_id, mat, vertices, faces in geometries:
        index_text = ' '.join(str(i) for face in faces for i in face)
        geometry_nodes.append(
            f'''    <geometry id="{geom_id}-geometry" name="{geom_id}">
      <mesh>
        <source id="{geom_id}-positions">
          <float_array id="{geom_id}-positions-array" count="{len(vertices) * 3}">{floats(vertices)}</float_array>
          <technique_common>
            <accessor source="#{geom_id}-positions-array" count="{len(vertices)}" stride="3">
              <param name="X" type="float"/>
              <param name="Y" type="float"/>
              <param name="Z" type="float"/>
            </accessor>
          </technique_common>
        </source>
        <vertices id="{geom_id}-vertices"><input semantic="POSITION" source="#{geom_id}-positions"/></vertices>
        <triangles material="{mat}" count="{len(faces)}">
          <input semantic="VERTEX" source="#{geom_id}-vertices" offset="0"/>
          <p>{index_text}</p>
        </triangles>
      </mesh>
    </geometry>'''
        )
        scene_nodes.append(
            f'''      <node id="{geom_id}" name="{geom_id}">
        <instance_geometry url="#{geom_id}-geometry">
          <bind_material><technique_common><instance_material symbol="{mat}" target="#{mat}-material"/></technique_common></bind_material>
        </instance_geometry>
      </node>'''
        )

    path.write_text(
        f'''<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset>
    <unit name="meter" meter="1"/>
    <up_axis>Z_UP</up_axis>
  </asset>
  <library_effects>
{chr(10).join(effects)}
  </library_effects>
  <library_materials>
{chr(10).join(material_nodes)}
  </library_materials>
  <library_geometries>
{chr(10).join(geometry_nodes)}
  </library_geometries>
  <library_visual_scenes>
    <visual_scene id="Scene" name="Scene">
{chr(10).join(scene_nodes)}
    </visual_scene>
  </library_visual_scenes>
  <scene><instance_visual_scene url="#Scene"/></scene>
</COLLADA>
''',
        encoding='utf-8',
    )


def write_mtl(path: Path, texture_rel: str, diffuse: tuple[float, float, float]) -> None:
    r, g, b = diffuse
    path.write_text(
        '\n'.join([
            'newmtl textured_surface',
            f'Ka {r:.3f} {g:.3f} {b:.3f}',
            f'Kd {r:.3f} {g:.3f} {b:.3f}',
            'Ks 0.120 0.120 0.120',
            'Ns 28.0',
            'illum 2',
            f'map_Kd {texture_rel}',
            '',
        ]),
        encoding='utf-8',
    )


def plastic_bag_texture(path: Path) -> None:
    w, h = 1024, 768
    img = Image.new('RGB', (w, h), (218, 226, 229))
    draw = ImageDraw.Draw(img, 'RGBA')

    shadow = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow, 'RGBA')
    sdraw.ellipse((150, 220, 900, 620), fill=(20, 30, 35, 70))
    shadow = shadow.filter(ImageFilter.GaussianBlur(32))
    img = Image.alpha_composite(img.convert('RGBA'), shadow)
    draw = ImageDraw.Draw(img, 'RGBA')

    bag = [(180, 260), (300, 170), (505, 210), (710, 150), (870, 300),
           (810, 570), (560, 650), (330, 610), (170, 470)]
    draw.polygon(bag, fill=(235, 245, 250, 235), outline=(150, 175, 188, 210))
    draw.polygon([(330, 610), (560, 650), (505, 210), (300, 170)],
                 fill=(210, 230, 240, 135))
    draw.polygon([(505, 210), (710, 150), (870, 300), (810, 570), (560, 650)],
                 fill=(245, 250, 252, 150))

    for offset, alpha in [(0, 150), (35, 120), (-45, 110), (80, 90)]:
        pts = []
        for i in range(9):
            x = 210 + i * 82
            y = 275 + 70 * math.sin(i * 1.35 + offset * 0.02) + offset
            pts.append((x, y))
        draw.line(pts, fill=(120, 150, 165, alpha), width=8)
        draw.line([(x, y - 8) for x, y in pts], fill=(255, 255, 255, 100), width=4)

    for x in (360, 650):
        draw.arc((x - 80, 105, x + 80, 305), start=190, end=350,
                 fill=(215, 232, 240, 230), width=18)
        draw.arc((x - 50, 130, x + 50, 280), start=190, end=350,
                 fill=(145, 170, 185, 125), width=4)

    for p in [(245, 360, 380, 455), (610, 395, 770, 500), (420, 500, 580, 590)]:
        draw.ellipse(p, outline=(255, 255, 255, 140), width=6)
    img.convert('RGB').save(path)


def _font(size: int):
    font_path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')
    if font_path.is_file():
        return ImageFont.truetype(str(font_path), size)
    return ImageFont.load_default()


def chip_bag_texture(path: Path) -> None:
    w, h = 1200, 780
    img = Image.new('RGB', (w, h), (196, 32, 35))
    draw = ImageDraw.Draw(img, 'RGBA')

    # Metallic packet folds.
    for x in range(-80, w + 120, 95):
        color = (255, 210, 80, 70) if (x // 95) % 2 == 0 else (70, 0, 0, 55)
        draw.polygon(
            [(x, 0), (x + 80, 0), (x + 35, h), (x - 45, h)],
            fill=color,
        )
    for y in range(50, h, 90):
        draw.line((0, y, w, y + 28 * math.sin(y * 0.05)),
                  fill=(255, 255, 255, 45), width=7)
        draw.line((0, y + 34, w, y + 34 + 20 * math.cos(y * 0.04)),
                  fill=(35, 0, 0, 50), width=5)

    # Sealed packet edges.
    draw.rectangle((24, 22, w - 24, h - 22), outline=(245, 210, 95, 230), width=18)
    draw.rectangle((64, 62, w - 64, h - 62), outline=(95, 10, 10, 145), width=5)
    for x in range(75, w - 75, 48):
        draw.line((x, 28, x + 16, 74), fill=(115, 35, 20, 120), width=4)
        draw.line((x, h - 28, x + 16, h - 74), fill=(115, 35, 20, 120), width=4)

    # Brand-like central label, kept generic.
    draw.ellipse((250, 120, 950, 610), fill=(255, 218, 68, 245),
                 outline=(105, 40, 18, 230), width=12)
    draw.ellipse((310, 170, 890, 560), outline=(255, 255, 255, 135), width=10)
    draw.text((365, 245), 'CHIPS', font=_font(128), fill=(115, 28, 18, 255),
              stroke_width=4, stroke_fill=(255, 245, 190, 255))
    draw.text((460, 388), 'CRISPY PACK', font=_font(34), fill=(85, 55, 20, 235))

    # Chips and small label details.
    chip_fill = (245, 190, 74, 245)
    for box, angle_line in [
        ((130, 530, 250, 620), (150, 585, 230, 555)),
        ((780, 455, 910, 555), (805, 505, 880, 530)),
        ((905, 260, 1025, 350), (920, 315, 1005, 285)),
        ((190, 185, 310, 275), (205, 235, 285, 205)),
    ]:
        draw.ellipse(box, fill=chip_fill, outline=(135, 75, 20, 160), width=5)
        draw.line(angle_line, fill=(160, 95, 30, 130), width=4)
    draw.rounded_rectangle((845, 600, 1080, 700), radius=10,
                           fill=(245, 245, 230, 230), outline=(60, 60, 60, 180), width=3)
    for i in range(5):
        y = 620 + i * 14
        draw.line((865, y, 1060, y), fill=(75, 75, 65, 150), width=3)

    # Transparent-looking dents and highlights.
    for box in [(90, 90, 240, 190), (990, 95, 1125, 210), (70, 640, 205, 735)]:
        draw.ellipse(box, outline=(255, 255, 255, 70), width=9)
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=145, threshold=3))
    img.save(path)


def write_model_sdf(model: str, collision_size: tuple[float, float, float],
                    mesh_ext: str = 'obj',
                    material_rgba: tuple[float, float, float, float] | None = None) -> None:
    sx, sy, sz = collision_size
    material = ''
    if material_rgba is not None:
        r, g, b, a = material_rgba
        material = f'''
        <material>
          <ambient>{r:.3f} {g:.3f} {b:.3f} {a:.3f}</ambient>
          <diffuse>{r:.3f} {g:.3f} {b:.3f} {a:.3f}</diffuse>
          <specular>0.08 0.08 0.08 1</specular>
        </material>'''
    (TRASH_ROOT / model / 'model.sdf').write_text(
        f'''<?xml version="1.0"?>
<sdf version="1.10">
  <model name="{model}">
    <static>true</static>
    <link name="link">
      <visual name="textured_mesh">
        <geometry>
          <mesh><uri>model://{model}/meshes/{model}.{mesh_ext}</uri></mesh>
        </geometry>
{material}
      </visual>
      <collision name="collision">
        <pose>0 0 {sz / 2:.4f} 0 0 0</pose>
        <geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry>
      </collision>
    </link>
  </model>
</sdf>
''',
        encoding='utf-8',
    )


def write_model_config(model: str, description: str) -> None:
    (TRASH_ROOT / model / 'model.config').write_text(
        f'''<?xml version="1.0"?>
<model>
  <name>{model}</name>
  <version>1.0</version>
  <sdf version="1.10">model.sdf</sdf>
  <author><name>trash_robot_sim procedural asset</name></author>
  <description>{description}</description>
</model>
''',
        encoding='utf-8',
    )


def generate_plastic_bag() -> None:
    mesh_dir, texture_dir = ensure_dirs('plastic_bag')
    plastic_bag_texture(texture_dir / 'plastic_bag_texture.png')
    write_wavy_plane_obj(mesh_dir / 'plastic_bag.obj', 'plastic_bag.mtl', 0.62, 0.45)
    write_mtl(mesh_dir / 'plastic_bag.mtl', '../textures/plastic_bag_texture.png', (0.86, 0.91, 0.94))
    if (mesh_dir / 'trash_bag_quaternius.glb').is_file():
        (TRASH_ROOT / 'plastic_bag' / 'model.sdf').write_text(
            '''<?xml version="1.0"?>
<sdf version="1.10">
  <model name="plastic_bag">
    <static>true</static>
    <link name="link">
      <visual name="trash_bag_mesh">
        <geometry>
          <mesh>
            <uri>model://plastic_bag/meshes/trash_bag_quaternius.glb</uri>
            <scale>1 1 1</scale>
          </mesh>
        </geometry>
      </visual>
      <collision name="collision">
        <pose>0 0 0.24 0 0 0</pose>
        <geometry><box><size>0.52 0.52 0.48</size></box></geometry>
      </collision>
    </link>
  </model>
</sdf>
''',
            encoding='utf-8',
        )
        write_model_config('plastic_bag', 'Downloaded realistic trash bag GLB mesh for plastic_bag demo class.')
    else:
        write_model_sdf('plastic_bag', (0.62, 0.45, 0.08))
        write_model_config('plastic_bag', 'Crumpled textured plastic bag OBJ mesh.')


def generate_paper_packaging() -> None:
    mesh_dir, texture_dir = ensure_dirs('paper_packaging')
    chip_bag_texture(texture_dir / 'paper_packaging_texture.png')
    write_chip_bag_dae(mesh_dir / 'paper_packaging.dae')
    write_wavy_plane_obj(
        mesh_dir / 'paper_packaging.obj',
        'paper_packaging.mtl',
        0.62,
        0.38,
        nx=20,
        ny=12,
        height=0.055,
    )
    write_mtl(mesh_dir / 'paper_packaging.mtl', '../textures/paper_packaging_texture.png', (0.88, 0.18, 0.14))
    write_model_sdf(
        'paper_packaging',
        (0.62, 0.38, 0.08),
        mesh_ext='dae',
        material_rgba=(0.86, 0.08, 0.04, 1.0),
    )
    write_model_config('paper_packaging', 'Crumpled colored chips packet DAE mesh for paper_packaging class.')


def main() -> None:
    generate_plastic_bag()
    generate_paper_packaging()
    print('Generated textured plastic_bag and chips-packet paper_packaging assets')


if __name__ == '__main__':
    main()
