import argparse
import csv
import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import open_clip
import PIL.Image as Image
import pyrender
import torch
import trimesh


CATEGORY_ALIASES = {
    "action change": "action_change",
    "action_change": "action_change",
    "object addition": "object_addition",
    "object_addition": "object_addition",
    "object removal": "object_removal",
    "object_removal": "object_removal",
    "object replacement": "object_replacement",
    "object_replacement": "object_replacement",
    "object style change": "object_style_change",
    "object_style_change": "object_style_change",
    "style change": "object_style_change",
    "style_change": "object_style_change",
}


def normalize_category(value: str) -> str:
    key = value.strip().lower().replace("-", "_")
    key = " ".join(key.replace("_", " ").split())
    return CATEGORY_ALIASES.get(key, key.replace(" ", "_"))


def _look_at(eye, target=(0, 0, 0), up=(0, 1, 0)):
    eye = np.asarray(eye)
    target = np.asarray(target)
    up = np.asarray(up)
    z = eye - target
    z = z / (np.linalg.norm(z) + 1e-8)
    x = np.cross(up, z)
    x = x / (np.linalg.norm(x) + 1e-8)
    y = np.cross(z, x)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, 0], transform[:3, 1], transform[:3, 2], transform[:3, 3] = x, y, z, eye
    return transform


def _make_turntable_views(n_views=6, elev_deg=20, radius=2.2):
    elev = np.deg2rad(elev_deg)
    for i in range(n_views):
        az = 2 * np.pi * i / n_views
        x = radius * np.cos(elev) * np.cos(az)
        y = radius * np.sin(elev)
        z = radius * np.cos(elev) * np.sin(az)
        yield np.array([x, y, z], dtype=np.float32)


def load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    return mesh


def render_multiview_images(
    mesh_path: Path,
    n_views: int = 6,
    elev_deg: float = 15.0,
    radius: float = 2.4,
    width: int = 512,
    height: int = 512,
):
    mesh = load_mesh(mesh_path)
    mesh.remove_unreferenced_vertices()
    mesh.apply_translation(-mesh.centroid)
    mesh.apply_scale(1.2 / max(mesh.extents.max(), 1e-6))

    scene = pyrender.Scene(
        bg_color=(255, 255, 255, 255),
        ambient_light=np.array([0.15, 0.15, 0.15, 1.0]),
    )
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(0.82, 0.82, 0.82, 1.0),
        metallicFactor=0.0,
        roughnessFactor=1.0,
    )
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True, material=material))

    cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(45.0))
    cam_node = scene.add(cam, pose=np.eye(4, dtype=np.float32))

    lights = []
    for eye, intensity in [
        ((2.5, 2.0, 1.5), 1.0),
        ((-2.5, 1.5, -0.5), 0.6),
        ((0.0, 2.5, -2.5), 0.8),
    ]:
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=intensity)
        lights.append(scene.add(light, pose=_look_at(eye)))

    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    images = []
    try:
        for eye in _make_turntable_views(n_views=n_views, elev_deg=elev_deg, radius=radius):
            scene.set_pose(cam_node, pose=_look_at(eye))
            color, _ = renderer.render(scene)
            images.append(Image.fromarray(color[..., :3]))
    finally:
        renderer.delete()
        for light_node in lights:
            try:
                scene.remove_node(light_node)
            except Exception:
                pass
    return images


def save_multiview_images(images, output_dir: Path, image_format: str, cols: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    ext = "jpg" if image_format == "jpeg" else image_format
    for i, image in enumerate(images):
        image.save(output_dir / f"view_{i:02d}.{ext}")
    if not images:
        return
    width, height = images[0].size
    rows = (len(images) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * width, rows * height), (255, 255, 255))
    for idx, image in enumerate(images):
        row, col = divmod(idx, cols)
        canvas.paste(image, (col * width, row * height))
    canvas.save(output_dir / f"sprite.{ext}")


def process_mesh_to_pointcloud(file_path: Path, n_points: int, device: str):
    try:
        mesh = load_mesh(file_path)
        points, _ = trimesh.sample.sample_surface(mesh, n_points)
    except Exception as exc:
        print(f"[ERROR] Failed to load or sample mesh {file_path}: {exc}")
        return None

    colors = np.ones((points.shape[0], 3), dtype=np.float32) * (100.0 / 255.0)
    feature = torch.cat(
        (torch.from_numpy(points).float(), torch.from_numpy(colors).float()),
        dim=-1,
    )
    return feature.unsqueeze(0).to(device)


def load_metadata(metadata_csv: Optional[str]) -> Tuple[Dict[Tuple[str, str], Dict[str, str]], Dict[str, Dict[str, str]]]:
    by_item: Dict[Tuple[str, str], Dict[str, str]] = {}
    by_sha: Dict[str, Dict[str, str]] = {}
    if not metadata_csv:
        return by_item, by_sha

    path = Path(metadata_csv)
    if not path.exists():
        raise FileNotFoundError(f"metadata CSV not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            raw_category = (
                row.get("category")
                or row.get("Category")
                or row.get("editing_category")
                or row.get("Editing_Category")
                or ""
            )
            category = normalize_category(raw_category)
            item_id = (
                row.get("item_id")
                or row.get("ItemID")
                or row.get("idx")
                or row.get("index")
                or str(row_idx % 20)
            )
            if category:
                by_item[(category, str(item_id))] = row
            sha = (row.get("sha256") or row.get("sha") or "").strip()
            if sha:
                by_sha[sha] = row
    return by_item, by_sha


def load_sha256_index_map(sha256_dir: Optional[str], categories: List[str]) -> Dict[str, Dict[int, str]]:
    if not sha256_dir:
        return {}
    sha_dir = Path(sha256_dir)
    mapping: Dict[str, Dict[int, str]] = {}
    for category in categories:
        fpath = sha_dir / f"{category}_sha256.txt"
        if not fpath.exists():
            print(f"[WARN] sha256 mapping missing for {category}: {fpath}")
            continue
        idx2sha = {}
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if ":" not in line:
                    continue
                left, right = line.strip().split(":", 1)
                try:
                    idx2sha[int(left.strip())] = right.strip()
                except ValueError:
                    continue
        mapping[category] = idx2sha
    return mapping


def get_target_text(row: Optional[Dict[str, str]]) -> str:
    if not row:
        return ""
    for key in ("target_text", "edited_text", "editing_instruction", "instruction"):
        value = (row.get(key) or "").strip()
        if value:
            return value
    return ""


def add_uni3d_repo_to_path(uni3d_repo: Optional[str]):
    if not uni3d_repo:
        return
    repo_path = Path(uni3d_repo).expanduser().resolve()
    if not repo_path.exists():
        raise FileNotFoundError(f"Uni3D repo not found: {repo_path}")
    sys.path.insert(0, str(repo_path))


def load_clip_model(cli_args, device: str):
    pretrained = None if cli_args.clip_pretrained.lower() in ("", "none") else cli_args.clip_pretrained
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        cli_args.clip_model_name,
        pretrained=pretrained,
    )

    if cli_args.clip_weights_path:
        weights_path = Path(cli_args.clip_weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"CLIP weights not found: {weights_path}")
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
        clip_model.load_state_dict(state_dict, strict=True)
    elif pretrained is None:
        print("[WARN] CLIP is initialized without pretrained weights. Pass --clip_pretrained or --clip_weights_path for evaluation.")

    clip_model.to(device).eval()
    return clip_model, clip_preprocess


def load_uni3d_model(cli_args, device: str):
    add_uni3d_repo_to_path(cli_args.uni3d_repo)
    try:
        import models.uni3d as uni3d_models
    except ImportError as exc:
        raise ImportError(
            "Cannot import Uni3D. Pass --uni3d_repo /path/to/Uni3D or run with PYTHONPATH pointing to the Uni3D repo."
        ) from exc

    model_configs = {
        "giant": {"pc_model": "eva_giant_patch14_560", "pc_feat_dim": 1408},
        "large": {"pc_model": "eva02_large_patch14_448", "pc_feat_dim": 1024},
        "base": {"pc_model": "eva02_base_patch14_448", "pc_feat_dim": 768},
        "small": {"pc_model": "eva02_small_patch14_224", "pc_feat_dim": 384},
        "tiny": {"pc_model": "eva02_tiny_patch14_224", "pc_feat_dim": 192},
    }
    config = model_configs[cli_args.model_size]
    model_args = Namespace(
        model="create_uni3d",
        npoints=cli_args.n_points,
        num_group=512,
        group_size=64,
        pc_encoder_dim=512,
        embed_dim=1024,
        pc_model=config["pc_model"],
        pc_feat_dim=config["pc_feat_dim"],
        ckpt_path=cli_args.ckpt_path,
        distributed=False,
        pretrained_pc=cli_args.pretrained_pc or "",
        drop_path_rate=0.0,
        patch_dropout=0.0,
    )

    model = getattr(uni3d_models, model_args.model)(args=model_args)
    checkpoint = torch.load(cli_args.ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("module", checkpoint)
    if next(iter(state_dict.keys())).startswith("module."):
        state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


def format_prediction_path(cli_args, category: str, item_id: str, idx: int, sha: Optional[str]) -> Path:
    assert cli_args.pred_root is not None
    values = {
        "category": category,
        "item_id": item_id,
        "idx": idx,
        "sha": sha or "",
        "sha8": (sha or "")[:8],
    }
    rel_dir = cli_args.pred_sample_dir_format.format(**values)
    pred_path = Path(cli_args.pred_root) / rel_dir / cli_args.pred_mesh_relpath
    if pred_path.exists() or sha:
        return pred_path

    category_dir = Path(cli_args.pred_root) / category
    candidates = sorted(category_dir.glob(f"{idx:02d}_*")) if category_dir.is_dir() else []
    if candidates:
        return candidates[0] / cli_args.pred_mesh_relpath
    return pred_path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate 3D editing results on Eval3DEdit with Uni3D and CLIP metrics.")
    parser.add_argument("testset_dir", type=str, help="Eval3DEdit directory with <category>/<item_id>/{src.glb,edited.png}.")
    parser.add_argument("--pred_root", type=str, default=None, help="Prediction root. If omitted, use <item>/output_edit/<edited_glb_name>.")
    parser.add_argument(
        "--pred_sample_dir_format",
        type=str,
        default="{category}/{item_id}",
        help="Prediction sample directory format relative to --pred_root. Placeholders: {category}, {item_id}, {idx}, {sha}, {sha8}.",
    )
    parser.add_argument("--pred_mesh_relpath", type=str, default="edited.glb", help="Predicted mesh path inside each prediction sample directory.")
    parser.add_argument("--edited_glb_name", type=str, default="edited.glb", help="Fallback mesh name under <item>/output_edit when --pred_root is omitted.")
    parser.add_argument("--metadata_csv", "--sha100_csv", dest="metadata_csv", type=str, default=None, help="Optional metadata CSV with target text.")
    parser.add_argument("--sha256_dir", type=str, default=None, help="Optional directory containing <category>_sha256.txt files.")
    parser.add_argument("--output_csv", type=str, default="results.csv", help="Output CSV path.")
    parser.add_argument("--max_samples", type=int, default=None, help="Only process first N valid samples.")

    parser.add_argument("--uni3d_repo", type=str, default=None, help="Path to the Uni3D repository, if it is not already on PYTHONPATH.")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to the Uni3D checkpoint.")
    parser.add_argument("--pretrained_pc", type=str, default=None, help="Optional Uni3D point-cloud encoder pretrained weights path.")
    parser.add_argument("--model_size", type=str, default="giant", choices=["giant", "large", "base", "small", "tiny"])
    parser.add_argument("--n_points", type=int, default=10000, help="Number of surface points sampled for Uni3D.")
    parser.add_argument("--clip_only", action="store_true", help="Skip Uni3D point-cloud metrics and only compute multiview CLIP metrics.")

    parser.add_argument("--clip_model_name", type=str, default="EVA02-E-14-plus", help="OpenCLIP model name.")
    parser.add_argument("--clip_pretrained", type=str, default="none", help="OpenCLIP pretrained tag, or 'none' when using --clip_weights_path.")
    parser.add_argument("--clip_weights_path", type=str, default=None, help="Optional local CLIP checkpoint path.")

    parser.add_argument("--n_views", type=int, default=6, help="Number of rendered multiview images.")
    parser.add_argument("--elev_deg", type=float, default=15.0, help="Camera elevation angle in degrees.")
    parser.add_argument("--radius", type=float, default=2.4, help="Camera radius.")
    parser.add_argument("--render_w", type=int, default=512, help="Render width.")
    parser.add_argument("--render_h", type=int, default=512, help="Render height.")
    parser.add_argument("--save_mv_dir", type=str, default=None, help="Optional directory for rendered multiview images.")
    parser.add_argument("--save_mv_cols", type=int, default=3, help="Columns in saved multiview sprite.")
    parser.add_argument("--save_mv_format", type=str, default="png", choices=["png", "jpg", "jpeg"])
    return parser.parse_args()


def main():
    cli_args = parse_args()
    if not cli_args.clip_only and not cli_args.ckpt_path:
        print("[ERROR] --ckpt_path is required unless --clip_only is set.")
        sys.exit(1)

    testset_dir = Path(cli_args.testset_dir)
    if not testset_dir.is_dir():
        print(f"[ERROR] Eval3DEdit directory not found: {testset_dir}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = load_clip_model(cli_args, device)
    uni3d_model = None if cli_args.clip_only else load_uni3d_model(cli_args, device)

    metadata_by_item, metadata_by_sha = load_metadata(cli_args.metadata_csv)
    category_paths = sorted([p for p in testset_dir.iterdir() if p.is_dir()])
    categories = [p.name for p in category_paths]
    category_idx2sha = load_sha256_index_map(cli_args.sha256_dir, categories)

    rows: List[List[Any]] = [[
        "Category",
        "ItemID",
        "Source_Mesh",
        "Pred_Mesh",
        "Target_Image",
        "PC_PC_Similarity",
        "PC_Image_Similarity",
        "PC_Text_Similarity",
        "MV_Text_Mean",
        "MV_Image_Mean",
    ]]

    processed = 0
    for category_path in category_paths:
        category = category_path.name
        item_paths = sorted(
            [p for p in category_path.iterdir() if p.is_dir()],
            key=lambda p: int(p.name) if p.name.isdigit() else p.name,
        )
        print(f"\nProcessing category: {category}")

        for item_path in item_paths:
            item_id = item_path.name
            if not item_id.isdigit():
                continue
            idx = int(item_id)
            source_mesh = item_path / "src.glb"
            target_image = item_path / "edited.png"
            sha = category_idx2sha.get(category, {}).get(idx)
            pred_mesh = (
                format_prediction_path(cli_args, category, item_id, idx, sha)
                if cli_args.pred_root
                else item_path / "output_edit" / cli_args.edited_glb_name
            )

            if not source_mesh.exists() or not target_image.exists():
                print(f"  [WARN] Skip {category}/{item_id}: missing src.glb or edited.png.")
                continue
            if not pred_mesh.exists():
                print(f"  [WARN] Skip {category}/{item_id}: predicted mesh not found: {pred_mesh}")
                continue

            item_row = metadata_by_item.get((category, item_id))
            if item_row is None and sha:
                item_row = metadata_by_sha.get(sha)
            target_text = get_target_text(item_row)

            try:
                similarity_pc_pc = None
                similarity_pc_img = None
                similarity_pc_txt = None
                pred_pc_emb = None

                target_img_tensor = clip_preprocess(Image.open(target_image).convert("RGB")).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_feature = clip_model.encode_image(target_img_tensor)
                    image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)

                if not cli_args.clip_only:
                    source_pc = process_mesh_to_pointcloud(source_mesh, cli_args.n_points, device)
                    pred_pc = process_mesh_to_pointcloud(pred_mesh, cli_args.n_points, device)
                    if source_pc is None or pred_pc is None:
                        continue
                    with torch.no_grad():
                        source_pc_emb = uni3d_model.encode_pc(source_pc)  # type: ignore[union-attr]
                        pred_pc_emb = uni3d_model.encode_pc(pred_pc)  # type: ignore[union-attr]
                        source_pc_emb = source_pc_emb / source_pc_emb.norm(dim=-1, keepdim=True)
                        pred_pc_emb = pred_pc_emb / pred_pc_emb.norm(dim=-1, keepdim=True)
                    similarity_pc_pc = torch.nn.functional.cosine_similarity(source_pc_emb, pred_pc_emb, dim=-1).item()
                    similarity_pc_img = torch.nn.functional.cosine_similarity(pred_pc_emb, image_feature, dim=-1).item()

                    if target_text:
                        with torch.no_grad():
                            text_tokens = open_clip.tokenize([target_text]).to(device)
                            text_feature = clip_model.encode_text(text_tokens)
                            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
                        similarity_pc_txt = torch.nn.functional.cosine_similarity(pred_pc_emb, text_feature, dim=-1).item()

                mv_text_mean = None
                mv_image_mean = None
                mv_images = render_multiview_images(
                    pred_mesh,
                    n_views=cli_args.n_views,
                    elev_deg=cli_args.elev_deg,
                    radius=cli_args.radius,
                    width=cli_args.render_w,
                    height=cli_args.render_h,
                )
                if cli_args.save_mv_dir:
                    save_multiview_images(
                        mv_images,
                        Path(cli_args.save_mv_dir) / category / item_id,
                        cli_args.save_mv_format,
                        max(1, cli_args.save_mv_cols),
                    )
                if mv_images:
                    with torch.no_grad():
                        batch_imgs = torch.stack([clip_preprocess(image) for image in mv_images], dim=0).to(device)
                        view_features = clip_model.encode_image(batch_imgs)
                        view_features = view_features / view_features.norm(dim=-1, keepdim=True)
                        mv_image_mean = torch.matmul(view_features, image_feature.T).squeeze(-1).mean().item()
                        if target_text:
                            text_tokens = open_clip.tokenize([target_text]).to(device)
                            text_feature = clip_model.encode_text(text_tokens)
                            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
                            mv_text_mean = torch.matmul(view_features, text_feature.T).squeeze(-1).mean().item()

                rows.append([
                    category,
                    item_id,
                    str(source_mesh),
                    str(pred_mesh),
                    str(target_image),
                    similarity_pc_pc if similarity_pc_pc is not None else "",
                    similarity_pc_img if similarity_pc_img is not None else "",
                    similarity_pc_txt if similarity_pc_txt is not None else "",
                    mv_text_mean if mv_text_mean is not None else "",
                    mv_image_mean if mv_image_mean is not None else "",
                ])
                processed += 1
                print(
                    f"  Processed {category}/{item_id}: "
                    f"MV-Text={mv_text_mean if mv_text_mean is not None else 'None'}, "
                    f"MV-Image={mv_image_mean if mv_image_mean is not None else 'None'}"
                )
                if cli_args.max_samples is not None and processed >= cli_args.max_samples:
                    break
            except Exception as exc:
                print(f"  [ERROR] Failed on {category}/{item_id}: {exc}")
        if cli_args.max_samples is not None and processed >= cli_args.max_samples:
            break

    output_csv = Path(cli_args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"\nDone. Processed {processed} samples. Results saved to {output_csv}")


if __name__ == "__main__":
    main()
