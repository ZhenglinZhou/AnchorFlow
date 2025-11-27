import os

os.environ['PYOPENGL_PLATFORM'] = 'egl'
import numpy as np
from pathlib import Path
from PIL import Image
import trimesh
import pyrender


def look_at(eye, target=(0, 0, 0), up=(0, 1, 0)):
    eye, target, up = map(np.asarray, (eye, target, up))
    z = eye - target;
    z = z / np.linalg.norm(z)
    x = np.cross(up, z);
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, eye
    return T


def make_turntable_views(n_views=6, elev_deg=20, radius=2.2):
    elev = np.deg2rad(elev_deg)
    for i in range(n_views):
        az = 2 * np.pi * i / n_views
        x = radius * np.cos(elev) * np.cos(az)
        y = radius * np.sin(elev)
        z = radius * np.cos(elev) * np.sin(az)
        yield np.array([x, y, z], dtype=np.float32)


def to_sprite(images, cols=3):
    if len(images) == 0: return None
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))
    for idx, im in enumerate(images):
        r, c = divmod(idx, cols)
        canvas.paste(im, (c * w, r * h))
    return canvas


def render_multiview(
        mesh_path,
        save_path,
        n_views=6,
        elev_deg=20,
        radius=2.2,
        w=640,
        h=640,
        bg=(255, 255, 255, 255),
):
    save_path = Path(save_path);

    mesh = trimesh.load(mesh_path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    mesh.remove_unreferenced_vertices()
    mesh.apply_translation(-mesh.centroid)
    scale = 1.0 / max(mesh.extents.max(), 1e-6)
    mesh.apply_scale(scale * 1.2)

    scene = pyrender.Scene(bg_color=bg, ambient_light=np.array([0.15, 0.15, 0.15, 1.0]))
    # pm = pyrender.Mesh.from_trimesh(mesh, smooth=True) # textured rendering
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(0.82, 0.82, 0.82, 1.0),
        metallicFactor=0.0,
        roughnessFactor=1.0
    )
    pm = pyrender.Mesh.from_trimesh(mesh, smooth=True, material=material)
    scene.add(pm)

    cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(45.0))
    cam_node = scene.add(cam, pose=np.eye(4))


    def add_dir_light(eye, intensity):
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=intensity)
        return scene.add(light, pose=look_at(eye))

    lights = [
        add_dir_light((2.5, 2.0, 1.5), 1.0),
        add_dir_light((-2.5, 1.5, -0.5), 0.6),
        add_dir_light((0.0, 2.5, -2.5), 0.8),
    ]

    r = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)

    images = []
    for i, eye in enumerate(make_turntable_views(n_views, elev_deg, radius)):
        pose = look_at(eye)
        scene.set_pose(cam_node, pose=pose)
        color, _ = r.render(scene)
        im = Image.fromarray(color[..., :3])
        # im_path = out_dir / f"view_{i:02d}.png"
        # im.save(im_path)
        images.append(im)

    sprite = to_sprite(images, cols=min(3, n_views))
    if sprite is not None:
        sprite_path = save_path
        sprite.save(sprite_path)
        print(f"[OK] save in {sprite_path}")
    r.delete()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--n_views", type=int, default=6)
    args = parser.parse_args()

    mesh_path = args.mesh_path
    save_path = args.save_path

    render_multiview(mesh_path, save_path=save_path, n_views=6, elev_deg=15, radius=2.4, w=1024, h=1024)
