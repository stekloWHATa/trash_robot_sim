#!/usr/bin/env python3
"""Generate local textured mesh assets for the Gazebo detection demo.

This is a deterministic fallback for machines without Blender installed.
The output is real OBJ/MTL + PNG texture, not SDF primitive blocks.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


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


def paper_texture(path: Path) -> None:
    w, h = 1024, 700
    img = Image.new('RGB', (w, h), (232, 224, 198))
    draw = ImageDraw.Draw(img, 'RGBA')
    draw.rectangle((55, 85, 970, 615), fill=(235, 196, 70, 255), outline=(95, 75, 35, 220), width=9)
    draw.polygon([(55, 85), (270, 110), (230, 615), (55, 615)], fill=(248, 248, 235, 245))
    draw.rectangle((730, 90, 845, 612), fill=(25, 130, 60, 245))
    draw.rectangle((855, 90, 970, 612), fill=(210, 50, 35, 230))
    draw.line((80, 240, 950, 205), fill=(120, 90, 45, 150), width=7)
    draw.line((110, 480, 920, 520), fill=(120, 90, 45, 120), width=5)
    draw.polygon([(760, 600), (970, 615), (900, 455)], fill=(180, 140, 55, 155))
    for x in range(120, 700, 90):
        draw.line((x, 100, x + 40, 610), fill=(255, 245, 170, 60), width=4)
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=3))
    img.save(path)


def write_model_sdf(model: str, collision_size: tuple[float, float, float]) -> None:
    sx, sy, sz = collision_size
    (TRASH_ROOT / model / 'model.sdf').write_text(
        f'''<?xml version="1.0"?>
<sdf version="1.10">
  <model name="{model}">
    <static>true</static>
    <link name="link">
      <visual name="textured_mesh">
        <geometry>
          <mesh><uri>model://{model}/meshes/{model}.obj</uri></mesh>
        </geometry>
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


def generate_plastic_bag() -> None:
    mesh_dir, texture_dir = ensure_dirs('plastic_bag')
    plastic_bag_texture(texture_dir / 'plastic_bag_texture.png')
    write_wavy_plane_obj(mesh_dir / 'plastic_bag.obj', 'plastic_bag.mtl', 0.62, 0.45)
    write_mtl(mesh_dir / 'plastic_bag.mtl', '../textures/plastic_bag_texture.png', (0.86, 0.91, 0.94))
    write_model_sdf('plastic_bag', (0.62, 0.45, 0.08))


def generate_paper_packaging() -> None:
    mesh_dir, texture_dir = ensure_dirs('paper_packaging')
    paper_texture(texture_dir / 'paper_packaging_texture.png')
    write_flat_box_obj(mesh_dir / 'paper_packaging.obj', 'paper_packaging.mtl', 0.42, 0.30, 0.035)
    write_mtl(mesh_dir / 'paper_packaging.mtl', '../textures/paper_packaging_texture.png', (0.92, 0.78, 0.34))
    write_model_sdf('paper_packaging', (0.42, 0.30, 0.04))


def main() -> None:
    generate_plastic_bag()
    generate_paper_packaging()
    print('Generated textured plastic_bag and paper_packaging assets')


if __name__ == '__main__':
    main()
