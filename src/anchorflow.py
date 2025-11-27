import os
import sys
from pathlib import Path

import math
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, 'hy3dshape')
sys.path.insert(0, 'hy3dpaint')

# ShapeVAE in Hunyuan3D
from hy3dshape.rembg import BackgroundRemover
from hy3dshape.surface_loaders import SharpEdgeSurfaceLoader
from hy3dshape.models.autoencoders import ShapeVAE
from hy3dshape.pipelines import export_to_trimesh

# ShapeDiT in Hunyuan3D
from hy3dshape.pipelines import retrieve_timesteps, Hunyuan3DDiTPipeline

try:
    from torchvision_fix import apply_fix

    apply_fix()
except ImportError:
    print("Warning: torchvision_fix module not found, proceeding without compatibility fix")
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")


class Hunyuan3DVAE:
    def __init__(
            self,
            model_path='tencent/Hunyuan3D-2.1',
            enable_flashvdm=False,
    ):
        self.vae = ShapeVAE.from_pretrained(
            model_path=model_path,
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
            octree_resolution=256,
            mc_algo='mc',
            enable_pbar=True
        )
        mesh = export_to_trimesh(mesh)[0]
        if save_path is not None:
            mesh.export(save_path)
            print(f"Successfully saved result to {save_path}")
        return mesh


def cosine_lambda_scheduler(current_step, total_steps, start=0.0, end=1.0):
    cos_inner = math.pi * min(current_step / total_steps, 1.0)
    return end - (end - start) * (0.5 * (1 + math.cos(cos_inner)))


class Hunyuan3DEdit(Hunyuan3DDiTPipeline):

    def make_condition_lat(self, image, guidance_scale, mask=None):
        do_classifier_free_guidance = guidance_scale >= 0 and not (
                hasattr(self.model, 'guidance_embed') and
                self.model.guidance_embed is True
        )
        cond_inputs = self.prepare_image(image, mask)
        image = cond_inputs.pop('image')
        cond = self.encode_cond(
            image=image,
            additional_cond_inputs=cond_inputs,
            do_classifier_free_guidance=do_classifier_free_guidance,
            dual_guidance=False,
        )
        return cond

    @staticmethod
    def _mix_cfg(cond: torch.Tensor, uncond: torch.Tensor, cfg: float) -> torch.Tensor:
        """Mixes conditional and unconditional predictions."""
        return uncond + cfg * (cond - uncond)

    def _calc_z_0(self, model_output, timestep, sample):
        t = timestep
        step_index = self.scheduler.index_for_timestep(t)
        current_sigma = self.scheduler.sigmas[step_index]
        x0 = sample - current_sigma * model_output
        return x0

    def _get_latent_anchor(self, model_output, timestep, sample):
        t = timestep
        step_index = self.scheduler.index_for_timestep(t)
        current_sigma = self.scheduler.sigmas[step_index]
        f_t = sample + (1 - current_sigma) * model_output  # Eq. (8) in main paper
        return f_t

    def _get_differential_edit_direction(self, t: torch.Tensor, zt_src: torch.Tensor,
                                         zt_tar: torch.Tensor, use_anchorflow=True) -> torch.Tensor:
        """Computes the differential edit direction (delta v) for a timestep."""

        _zt_src = torch.cat([zt_src] * 2)
        _zt_tar = torch.cat([zt_tar] * 2)

        B = _zt_src.shape[0]
        t_batch = (t if isinstance(t, torch.Tensor) else torch.tensor(t, device=_zt_src.device))
        t_batch = t_batch.to(device=_zt_src.device, dtype=torch.float32).repeat(B)
        t_batch = t_batch / self.scheduler.config.num_train_timesteps

        vt_src_pred = self.model(_zt_src, t_batch, self.src_cond_lat, guidance=self.src_guidance)
        vt_src_cond, vt_src_uncond = vt_src_pred.chunk(2)
        vt_src = vt_src_uncond + self.src_guidance_scale * (vt_src_cond - vt_src_uncond)

        vt_tar_pred = self.model(_zt_tar, t_batch, self.tar_cond_lat, guidance=self.tar_guidance)
        vt_tar_cond, vt_tar_uncond = vt_tar_pred.chunk(2)
        vt_tar = vt_tar_uncond + self.tar_guidance_scale * (vt_tar_cond - vt_tar_uncond)

        output = vt_tar - vt_src

        if use_anchorflow:
            step_index = self.scheduler.index_for_timestep(t)
            current_sigma = self.scheduler.sigmas[step_index]

            ft_src = self._get_latent_anchor(zt_src, t, vt_src)
            ft_tar = self._get_latent_anchor(zt_tar, t, vt_tar)
            output = (2 - current_sigma) * (ft_tar - ft_src)  # Eq.(11) in main paper
        return output

    def _propagate_for_timestep(self, zt_inv: torch.Tensor, t: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        B = zt_inv.shape[0]
        t_batch = (t if isinstance(t, torch.Tensor) else torch.tensor(t, device=zt_inv.device))
        t_batch = t_batch.to(device=zt_inv.device, dtype=torch.float32).repeat(B)

        n_avg = self.n_avg
        diff_v_avg = 0
        for _ in range(n_avg):
            if self.anchor_noise:
                fwd_noise = self.fwd_noise
            else:
                fwd_noise = torch.randn_like(self.x_src)
            zt_src = self.scheduler.scale_noise(self.x_src, t_batch, fwd_noise)

            # EditP23
            # zt_tar = self.scheduler.scale_noise(zt_inv, t_batch, fwd_noise)

            # FlowEdit
            zt_tar = zt_inv + zt_src - self.x_src

            # Inference
            # zt_tar = zt_inv
            diff_v = self._get_differential_edit_direction(t, zt_src, zt_tar, use_anchorflow=self.use_anchorflow)
            diff_v_avg += diff_v
        diff_v = diff_v_avg / n_avg
        zt_inv_change = dt * diff_v
        zt_inv = zt_inv.to(torch.float32) + zt_inv_change
        return zt_inv.to(diff_v.dtype)

    def _get_guidance(self, guidance_scale, batch_size, device, dtype):
        guidance = None
        if hasattr(self.model, 'guidance_embed') and \
                self.model.guidance_embed is True:
            guidance = torch.tensor([guidance_scale] * batch_size, device=device, dtype=dtype)
        return guidance

    def inversion(self, latents, inv_timesteps, inv_sigmas, inv_guidance_scale=1.0):
        inv_latent = None
        inv_latents = {}
        sigmas = inv_sigmas
        for i, t in enumerate(tqdm(inv_timesteps, desc="FM Inverse Sampling:")):
            latent_model_input = torch.cat([latents] * 2)
            timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)
            timestep = timestep / self.scheduler.config.num_train_timesteps
            vt_pred = self.model(latent_model_input, timestep, self.src_cond_lat, guidance=self.src_guidance)
            vt_cond, vt_uncond = vt_pred.chunk(2)
            vt = vt_uncond + inv_guidance_scale * (vt_cond - vt_uncond)

            latents = latents.to(torch.float32)
            if inv_latent is None:
                sigma = sigmas[i]
                sigma_next = sigmas[i + 1]
                dt = sigma_next - sigma
                inv_latent = latents
                latents = latents + dt * vt
            else:
                sigma = sigmas[i - 1]
                sigma_next = sigmas[i]
                dt = sigma_next - sigma
                inv_latent = inv_latent + dt * vt
                if (i + 1) < len(sigmas):
                    sigma_next_next = sigmas[i + 1]
                    dt_next = sigma_next_next - sigma_next
                    latents = inv_latent + dt_next * vt
                else:
                    latents = inv_latent
            inv_latents[f'{int(t)}'] = inv_latent.clone()
            latents = latents.to(vt.dtype)
        return latents, inv_latents

    def inference(self, latents, timesteps, guidance_scale=5.0):
        sigmas = self.scheduler.sigmas
        for i, t in enumerate(tqdm(timesteps, desc="FM Sampling:")):
            latent_model_input = torch.cat([latents] * 2)
            timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)
            timestep = timestep / self.scheduler.config.num_train_timesteps
            vt_pred = self.model(latent_model_input, timestep, self.src_cond_lat, guidance=self.src_guidance)
            vt_cond, vt_uncond = vt_pred.chunk(2)
            vt = vt_uncond + guidance_scale * (vt_cond - vt_uncond)

            latents = latents.to(torch.float32)
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            dt = sigma_next - sigma
            latents = latents + dt * vt
            latents = latents.to(vt.dtype)
        return latents

    @torch.no_grad()
    def denoise(self, x_src, src_cond_img, tar_cond_img, edit_kwargs):
        # editing kwargs
        self.T_steps = edit_kwargs.get('T_steps', 50)
        self.src_guidance_scale = edit_kwargs.get('src_guidance_scale', 3.5)
        self.tar_guidance_scale = edit_kwargs.get('tar_guidance_scale', 5.0)
        self.n_avg = edit_kwargs.get('n_avg', 1)
        self.n_max = edit_kwargs.get('n_max', 31)
        self.anchor_noise = edit_kwargs.get('anchor_noise', False)
        self.w_inversion = edit_kwargs.get('inversion', False)
        self.use_anchorflow = edit_kwargs.get('use_anchorflow', True)
        inference = edit_kwargs.get('infer', False)

        # flow matching params
        self.x_src = x_src
        self.src_cond_img, self.tar_cond_img = src_cond_img, tar_cond_img

        # model params
        device, dtype = self.device, self.dtype
        batch_size = x_src.shape[0]

        # condition latents
        self.src_guidance = self._get_guidance(self.src_guidance_scale, batch_size, device, dtype)
        self.tar_guidance = self._get_guidance(self.tar_guidance_scale, batch_size, device, dtype)
        self.src_cond_lat = self.make_condition_lat(self.src_cond_img, guidance_scale=self.src_guidance_scale)
        self.tar_cond_lat = self.make_condition_lat(self.tar_cond_img, guidance_scale=self.tar_guidance_scale)

        # prepare timesteps
        sigmas = np.linspace(0, 1, self.T_steps)
        timesteps, _ = retrieve_timesteps(
            self.scheduler,
            self.T_steps,
            device,
            sigmas=sigmas,
        )

        # inverse timesteps
        if self.w_inversion:
            inv_timesteps = timesteps.flip(dims=[0]).to(device)
            inv_sigmas = self.scheduler.sigmas.flip(dims=[0]).to(device)

            # zt_inv = self.prepare_latents(batch_size=batch_size, dtype=dtype, device=device, generator=None) # debug
            zt_inv, zt_inv_list = self.inversion(
                self.x_src, inv_timesteps=inv_timesteps,
                inv_sigmas=inv_sigmas, inv_guidance_scale=1.0
            )
            self.fwd_noise = zt_inv.to(self.x_src)
        else:
            self.fwd_noise = torch.randn_like(self.x_src)

        # editing
        zt_inv_list = []
        if inference:
            zt_inv = self.inference(self.fwd_noise, timesteps=timesteps, guidance_scale=5.0)  # debug
        else:
            zt_inv = self.x_src.clone()
            start_index = max(0, len(timesteps) - self.n_max)
            for i in tqdm(range(start_index, len(timesteps) - 1)):
                self.n = i
                t = timesteps[i]
                t_i = t / self.scheduler.config.num_train_timesteps

                t_im1 = timesteps[i + 1]
                t_im1 = t_im1 / self.scheduler.config.num_train_timesteps
                dt = t_im1 - t_i
                zt_inv = self._propagate_for_timestep(zt_inv, t, dt)
                zt_inv_list.append(zt_inv.clone())
        return zt_inv, zt_inv_list


def load_image(image_path, save_path=None):
    if save_path and os.path.exists(save_path):
        image = Image.open(save_path).convert('RGBA')
        print(f"Successfully loaded image from {save_path}")
    else:
        image = Image.open(image_path).convert("RGBA")
        rembg = BackgroundRemover()
        image = rembg(image)
        if save_path is not None:
            image.save(save_path)
            print(f"Successfully saved preprocessed image to {save_path}")
    return image


def edit_3d_model(
        src_img_path,
        tar_img_path,
        src_mesh,
        n_max=31,
        src_guidance_scale=3.5,
        tar_guidance_scale=5.0,
        use_anchorflow=True,
        anchor_noise=False,
        inversion=False,
        infer=False,
        output_path='examples/1/edited.glb',
        vis_trajectory=False,
):
    src_img = load_image(src_img_path, src_img_path.replace(".png", "_rm_bg.png"))
    tar_img = load_image(tar_img_path, tar_img_path.replace(".png", "_rm_bg.png"))

    latents = hunyuan_vae.encode(src_mesh)
    edit_latents, edit_trajectory = hunyuan_3dedit.denoise(latents, src_img, tar_img, {
        'T_steps': 50,
        'n_max': n_max,
        'src_guidance_scale': src_guidance_scale,
        'tar_guidance_scale': tar_guidance_scale,
        'use_anchorflow': use_anchorflow,
        'anchor_noise': anchor_noise,
        'inversion': inversion,
        'infer': infer,
    })
    if vis_trajectory:
        output_path = Path(output_path)
        out_dir = output_path.parent / output_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        for i, latent in enumerate(edit_trajectory):
            name = f"{i}.glb"
            save_path = out_dir / name
            hunyuan_vae.decode(latent, save_path=str(save_path))
            saved.append(save_path)
    else:
        hunyuan_vae.decode(edit_latents, save_path=output_path)


def set_seed(seed=42):
    import os, random, numpy as np, torch
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    # load from huggingface
    model_path = "tencent/Hunyuan3D-2.1"

    # load from local
    # os.environ["HY3DGEN_MODELS"] = '/path/to/the/hunyuan3d/model_weights'
    # model_path = 'Hunyuan3D-2.1'

    hunyuan_vae = Hunyuan3DVAE(model_path=model_path)
    hunyuan_3dedit = Hunyuan3DEdit.from_pretrained(model_path=model_path)

    # (n_max, tgs): (35, 5.0), (37, 6.0), (41, 7.5), (45, 10.0)
    seed = 42
    n_max = 41
    tar_guidance_scale = 7.5
    use_anchorflow = True
    tag = "anchorflow" if use_anchorflow else "baseline"

    exp_dir = Path("examples/0")
    output_dir = exp_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    src_img_path = exp_dir / "src.png"
    tar_img_path = exp_dir / "edited.png"
    src_mesh = exp_dir / "src.glb"

    output_path = output_dir / f"{tag}_tgs_{tar_guidance_scale}_nmax_{n_max}.glb"
    debug_views = output_dir / f"{tag}_tgs_{tar_guidance_scale}_nmax_{n_max}.png"

    edit_3d_model(
        src_img_path=str(src_img_path),
        tar_img_path=str(tar_img_path),
        src_mesh=str(src_mesh),
        n_max=n_max,
        src_guidance_scale=3.5,
        tar_guidance_scale=tar_guidance_scale,
        use_anchorflow=use_anchorflow,
        anchor_noise=False,
        inversion=False,
        output_path=str(output_path),
        vis_trajectory=False,
        infer=False,
    )

    os.system(
        f"python3 src/rendering/mesh_render.py "
        f"--mesh_path {str(output_path)} "
        f"--save_path {str(debug_views)}"
    )
