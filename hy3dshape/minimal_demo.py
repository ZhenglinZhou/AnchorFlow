# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import os
from pathlib import Path

import torch

from hy3dshape.inv_schedulers import UniInvEulerScheduler
from hy3dshape.models.autoencoders import ShapeVAE
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from hy3dshape.pipelines import export_to_trimesh
from hy3dshape.surface_loaders import SharpEdgeSurfaceLoader


class Hunyuan3DVAE:
    def __init__(self, enable_flashvdm=True):
        self.vae = ShapeVAE.from_pretrained(
            'tencent/Hunyuan3D-2.1',
            use_safetensors=False,
            variant='fp16',
        )
        if enable_flashvdm:
            self.vae.enable_flashvdm_decoder(
                enabled=True,
                adaptive_kv_selection=True,
                topk_mode='mean',
                mc_algo='mc'
            )

        self.loader = SharpEdgeSurfaceLoader(
            num_sharp_points=0,
            num_uniform_points=81920,
        )

    def encode(self, mesh_path):
        surface = self.loader(mesh_path).to('cuda', dtype=torch.float16)
        print(surface.shape)
        latents = self.vae.encode(surface)
        return latents

    @torch.no_grad()
    def decode(self, latents, save_path=None):
        latents = self.vae.decode(latents)
        mesh = self.vae.latents2mesh(
            latents,
            output_type='trimesh',
            bounds=1.01,
            mc_level=0.0,
            num_chunks=20000,
            octree_resoluton=256,
            mc_algo='mc',
            enable_pbar=True
        )
        mesh = export_to_trimesh(mesh)[0]
        if save_path is not None:
            mesh.export(save_path)
            print(f"Successfully saved result to {save_path}")
        return mesh


FOLDERS = [
    "action_change",
    "object_addition",
    "object_removal",
    "object_replacement",
    "object_style_change",
]

if __name__ == '__main__':
    model_path = 'tencent/Hunyuan3D-2.1'
    hunyuan_vae = Hunyuan3DVAE()

    pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)

    scheduler = pipeline_shapegen.scheduler
    inv_scheduler = UniInvEulerScheduler.from_config(scheduler.config)

    folder = "action_change"
    for idx in range(20):
        exp_dir = f"frontal_view_models/{folder}/{idx}"
        exp_dir = Path(exp_dir)

        # src_img_path = exp_dir / "src.png"
        # tar_img_path = exp_dir / "edited.png"
        # src_mesh = "examples/teaser/prompt1/src.glb"
        src_mesh = exp_dir / "src.glb"
        latents = hunyuan_vae.encode(str(src_mesh))

        output_dir = exp_dir / "output_edit"
        # output_dir = exp_dir / "ablation_study"
        output_dir.mkdir(exist_ok=True)
        # output_path = output_dir / f"result_sgs_{src_guidance_scale}_tgs_{tar_guidance_scale}_nmax_{n_max}_reg_{reg}.glb"
        # debug_views = output_dir / f"result_sgs_{src_guidance_scale}_tgs_{tar_guidance_scale}_nmax_{n_max}_reg_{reg}.png"

        output_path = output_dir / "reconstruct.glb"
        debug_views = output_dir / "reconstruct.png"

        recon_mesh = hunyuan_vae.decode(latents, str(output_path))

        os.system(
            f"python3 src/rendering/mesh_render.py --mesh_path {str(output_path)} --save_path {str(debug_views)}")
