# coding=utf-8
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import math
import numpy as np
import torch
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler, DDIMInverseScheduler
from diffusers.utils import BaseOutput


@dataclass
class UniInvEulerSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor


@dataclass
class UniInvDDIMSchedulerOutput(BaseOutput):
    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


class UniInvEulerScheduler(FlowMatchEulerDiscreteScheduler):
    zero_initial = False
    alpha = 1

    def set_hyperparameters(self, zero_initial=False, alpha=1):
        self.zero_initial = zero_initial
        self.alpha = alpha

    def set_timesteps(
            self,
            num_inference_steps: int = None,
            device: Union[str, torch.device] = None,
            sigmas: Optional[List[float]] = None,
            mu: Optional[float] = None,
    ):
        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError(" you have a pass a value for `mu` when `use_dynamic_shifting` is set to be `True`")

        if sigmas is None:
            self.num_inference_steps = num_inference_steps
            timesteps = np.linspace(
                self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps
            )

            sigmas = timesteps / self.config.num_train_timesteps
        else:
            self.num_inference_steps = len(sigmas)

        if self.config.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            sigmas = self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas)

        # if self.config.use_karras_sigmas:
        #     sigmas = self._convert_to_karras(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
        #
        # elif self.config.use_exponential_sigmas:
        #     sigmas = self._convert_to_exponential(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
        #
        # elif self.config.use_beta_sigmas:
        #     sigmas = self._convert_to_beta(in_sigmas=sigmas, num_inference_steps=num_inference_steps)

        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)

        # timesteps
        # NOTE: this is slightly different from common usage, Hunyuan3D start from 0.
        timesteps = sigmas * self.config.num_train_timesteps
        # timesteps = torch.cat([timesteps, torch.zeros(1).to(sigmas)])
        timesteps = torch.cat([timesteps, torch.ones(1).to(sigmas)])
        self.timesteps = timesteps.flip(dims=[0]).to(device=device)

        # sigmas
        # sigmas = torch.cat([sigmas, torch.zeros(1).to(sigmas)])
        sigmas = torch.cat([sigmas, torch.ones(1, device=sigmas.device)])
        self.sigmas = sigmas.flip(dims=[0]).to(device=device)

        # empty dt and derivative
        self.sample = None

        # zero_initial
        if self.zero_initial:
            self.timesteps = self.timesteps[1:]
            self.sigmas = self.sigmas[1:]
            self.sample = 'placeholder'
            self.first_sigma = 0

        # alpha, early stop
        if self.alpha < 1:
            inv_steps = math.floor(self.alpha * self.num_inference_steps)
            skip_steps = self.num_inference_steps - inv_steps
            self.timesteps = self.timesteps[: -skip_steps]
            self.sigmas = self.sigmas[: -skip_steps]

        self._step_index = 0
        self._begin_index = 0

    def step(
            self,
            model_output: torch.FloatTensor,
            timestep: Union[float, torch.FloatTensor],
            sample: torch.FloatTensor,
            s_churn: float = 0.0,
            s_tmin: float = 0.0,
            s_tmax: float = float("inf"),
            s_noise: float = 1.0,
            generator: Optional[torch.Generator] = None,
            return_dict: bool = True,
    ) -> Union[UniInvEulerSchedulerOutput, Tuple]:

        if (
                isinstance(timestep, int)
                or isinstance(timestep, torch.IntTensor)
                or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `HeunDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        sample = sample.to(torch.float32)

        if self.sample is None:
            # just for the first step
            sigma = self.sigmas[self.step_index]
            sigma_next = self.sigmas[self.step_index + 1]

            derivative = model_output  # v_0 = f(t=0, x_0)
            dt = sigma_next - sigma  # sigma_{t + \Delta t} - sigma_t

            # store for correction
            self.sample = sample  # Z_0

            prev_sample = sample + derivative * dt
            prev_sample = prev_sample.to(model_output.dtype)
        else:
            sigma = self.sigmas[self.step_index - 1]
            sigma_next = self.sigmas[self.step_index]

            if isinstance(self.sample, str):
                # for zero_initial
                sigma = self.first_sigma
                self.sample = sample

            derivative = model_output
            dt = sigma_next - sigma

            sample = self.sample

            self.sample = sample + dt * derivative

            if (self.step_index + 1) < len(self.sigmas):
                sigma_next_next = self.sigmas[self.step_index + 1]
                dt_next = sigma_next_next - sigma_next

                prev_sample = self.sample + dt_next * derivative
            else:
                # end loop
                prev_sample = self.sample
            prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return UniInvEulerSchedulerOutput(prev_sample=prev_sample)


class UniInvDDIMScheduler(DDIMInverseScheduler):
    min_noise_alpha_cumprod = torch.tensor(1.0)
    max_noise_alpha_cumprod = torch.tensor(0.0)
    zero_init = False
    alpha = 1

    def set_hyperparameters(self, zero_initial=False, alpha=1):
        self.zero_initial = zero_initial
        self.alpha = alpha

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        if num_inference_steps > self.config.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
                f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                f" maximal {self.config.num_train_timesteps} timesteps."
            )

        self.num_inference_steps = num_inference_steps

        # "leading" and "trailing" corresponds to annotation of Table 1. of https://arxiv.org/abs/2305.08891
        if self.config.timestep_spacing == "leading":
            step_ratio = self.config.num_train_timesteps // self.num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round().copy().astype(np.int64)
            timesteps += self.config.steps_offset
        elif self.config.timestep_spacing == "trailing":
            step_ratio = self.config.num_train_timesteps / self.num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = np.round(np.arange(self.config.num_train_timesteps, 0, -step_ratio)[::-1]).astype(np.int64)
            timesteps -= 1
        else:
            raise ValueError(
                f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'leading' or 'trailing'."
            )

        self.timesteps = torch.from_numpy(timesteps).to(device)

        self.sample = None
        self.timesteps = torch.cat([torch.zeros(1).to(self.timesteps), self.timesteps])

        if self.zero_init:
            self.timesteps = self.timesteps[1:]

        # alpha, early stop
        if self.alpha < 1:
            inv_steps = math.floor(self.alpha * self.num_inference_steps)
            skip_steps = self.num_inference_steps - inv_steps
            self.timesteps = self.timesteps[: -skip_steps]

    def get_sample_epsilon(self, alpha_prod_t, sample, model_output, custom_prediction_type=None):
        beta_prod_t = 1 - alpha_prod_t
        prediction_type = (
            self.config.prediction_type if custom_prediction_type is None
            else custom_prediction_type
        )

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        elif prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
        elif prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t ** 0.5) * sample - (beta_prod_t ** 0.5) * model_output
            pred_epsilon = (alpha_prod_t ** 0.5) * model_output + (beta_prod_t ** 0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )
        # 4. Clip or threshold "predicted x_0"
        if self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        return pred_original_sample, pred_epsilon

    def step(
            self,
            model_output: torch.Tensor,
            timestep: int,
            sample: torch.Tensor,
            return_dict: bool = True,
    ) -> Union[UniInvDDIMSchedulerOutput, Tuple]:

        flag_init = False
        if self.sample is None:
            flag_init = True
            self.sample = sample

        sample = self.sample

        # 1. get previous step value
        prev_timestep = timestep
        timestep = min(
            timestep - self.config.num_train_timesteps // self.num_inference_steps, self.config.num_train_timesteps - 1
        )

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep] if timestep >= 0 else self.min_noise_alpha_cumprod
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep]

        # 3. compute predicted original sample from predicted noise also called
        # 4. Clip or threshold "predicted x_0"
        pred_original_sample, pred_epsilon = self.get_sample_epsilon(alpha_prod_t, sample, model_output)

        # 5. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon

        # 6. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        if (not flag_init) or self.zero_init:
            self.sample = prev_sample

            if prev_timestep < self.timesteps[-1]:
                prev_prev_sample = prev_timestep + self.config.num_train_timesteps // self.num_inference_steps
                alpha_prod_t_prev_prev = self.alphas_cumprod[prev_prev_sample] \
                    if prev_prev_sample < self.config.num_train_timesteps \
                    else self.alphas_cumprod[-1]

                prev_sample = alpha_prod_t_prev_prev ** (0.5) * pred_original_sample + \
                              (1 - alpha_prod_t_prev_prev) ** (0.5) * pred_epsilon

            else:
                # end loop
                prev_sample = self.sample

        if not return_dict:
            return (prev_sample, pred_original_sample)
        return UniInvDDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)
