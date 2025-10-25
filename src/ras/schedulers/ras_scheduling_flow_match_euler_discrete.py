# This file is a modified version of the original file from the HuggingFace/diffusers library.

# Copyright 2024 Stability AI, Katherine Crowson and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
from datetime import datetime
import torch.nn.functional as F 
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, logging
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from ras.utils import ras_manager


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class RASFlowMatchEulerDiscreteSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor


class RASFlowMatchEulerDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
    """
    RAS Euler scheduler.

    This model inherits from ['FlowMatchEulerDiscreteScheduler']. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        timestep_spacing (`str`, defaults to `"linspace"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        shift (`float`, defaults to 1.0):
            The shift value for the timestep schedule.
    """

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        use_dynamic_shifting=False,
        base_shift: Optional[float] = 0.5,
        max_shift: Optional[float] = 1.15,
        base_image_seq_len: Optional[int] = 256,
        max_image_seq_len: Optional[int] = 4096,
        invert_sigmas: bool = False,
    ):
        super().__init__(num_train_timesteps=num_train_timesteps,
                         shift=shift,
                         use_dynamic_shifting=use_dynamic_shifting,
                         base_shift=base_shift,
                         max_shift=max_shift,
                         base_image_seq_len=base_image_seq_len,
                         max_image_seq_len=max_image_seq_len,
                        #  invert_sigmas=invert_sigmas
                         )
        self.drop_cnt = None
        # code for experimentation evaluation
        if ras_manager.MANAGER.std_experiment:
            self.std_history = None
            self.mean_history = None
            self.prevDiff = None
            self.dot_history = None


    def _init_ras_config(self, latents):
        self.drop_cnt = torch.zeros((latents.shape[-2] // ras_manager.MANAGER.patch_size * latents.shape[-1] // ras_manager.MANAGER.patch_size), device=latents.device) - len(self.sigmas)

        # code for experimentation evaluation        
        if ras_manager.MANAGER.std_experiment:
            self.std_history = []
            self.mean_history = []
            self.prevDiff = None
            self.dot_history = []


    def extract_latents_index_from_patched_latents_index(self, indices, height):
        # # TODO add non-square case
        # # TODO support PATCH_SIZE != 2
        flattened_indices = indices // (height // ras_manager.MANAGER.patch_size) * ras_manager.MANAGER.patch_size * height + indices % (height // ras_manager.MANAGER.patch_size) *ras_manager.MANAGER.patch_size
        flattened_indices = (flattened_indices[:, None] + torch.tensor([0, height + 1, 1, height], dtype=indices.dtype, device=indices.device)[None, :]).flatten()
        return flattened_indices

    def ras_selection(self, sample, diff, height, width):
        diff = diff.squeeze(0).permute(1, 2, 0)
        # calculate the metric for each patch
        if ras_manager.MANAGER.metric == "std":
            metric = torch.std(diff, dim=-1).view(height // ras_manager.MANAGER.patch_size, ras_manager.MANAGER.patch_size, width // ras_manager.MANAGER.patch_size, ras_manager.MANAGER.patch_size).transpose(-2, -3).mean(-1).mean(-1).view(-1)
        elif ras_manager.MANAGER.metric == "l2norm":
            metric = torch.norm(diff, p=2, dim=-1).view(height // ras_manager.MANAGER.patch_size, ras_manager.MANAGER.patch_size, width // ras_manager.MANAGER.patch_size, ras_manager.MANAGER.patch_size).transpose(-2, -3).mean(-1).mean(-1).view(-1)
        else:
            raise ValueError("Unknown metric")

        # scale the metric with the drop count to avoid starvation
        metric *= torch.exp(ras_manager.MANAGER.starvation_scale * self.drop_cnt)
        current_skip_num = ras_manager.MANAGER.skip_token_num_list[self._step_index + 1]
        assert ras_manager.MANAGER.high_ratio >= 0 and ras_manager.MANAGER.high_ratio <= 1, "High ratio should be in the range of [0, 1]"
        indices = torch.sort(metric, dim=0, descending=False).indices
        low_bar = int(current_skip_num * (1 - ras_manager.MANAGER.high_ratio))
        high_bar = int(current_skip_num * ras_manager.MANAGER.high_ratio)
        cached_patchified_indices = torch.cat([indices[:low_bar], indices[-high_bar:]])
        other_patchified_indices = indices[low_bar:-high_bar]
        self.drop_cnt[cached_patchified_indices] += 1
        latent_cached_indices = self.extract_latents_index_from_patched_latents_index(cached_patchified_indices, height)

        return latent_cached_indices, other_patchified_indices

    def _save_analysis_data(self):
        """
        Stacks the collected std_history and saves the relevant tensors.
        """
        if not self.std_history:
            logger.info("std_history is empty. No data to save.")
            return

        # 1. Create a unique folder for this run's output
        folder_name = f"std_analysis_{ras_manager.MANAGER.name_folder}"
        os.makedirs(folder_name, exist_ok=True)
        logger.info(f"Saving analysis data to folder: {folder_name}")

        # 2. Stack and save the standard deviation history
        stacked_history = torch.stack(self.std_history)
        history_path = os.path.join(folder_name, 'std_history.pt')
        torch.save(stacked_history, history_path)
        print(stacked_history.shape)
        logger.info(f"Saved stacked std tensor to {history_path}")

        stacked_history = torch.stack(self.mean_history)
        history_path = os.path.join(folder_name, 'mean_history.pt')
        torch.save(stacked_history, history_path)
        print(stacked_history.shape)
        logger.info(f"Saved stacked std tensor to {history_path}")

        dot_tensor = torch.stack(self.dot_history, dim=0)
        dot_path = os.path.join(folder_name, 'dot_history.pt')
        torch.save(dot_tensor, dot_path)
        # The shape will be (num_steps, num_patches)
        print(f"Dot history shape: {dot_tensor.shape}")
        logger.info(f"Saved dot product history tensor to {dot_path}")


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
    ) -> Union[RASFlowMatchEulerDiscreteSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            s_churn (`float`):
            s_tmin  (`float`):
            s_tmax  (`float`):
            s_noise (`float`, defaults to 1.0):
                Scaling factor for noise added to the sample.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or
                tuple.

        Returns:
            [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] is
                returned, otherwise a tuple is returned where the first element is the sample tensor.
        """
        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )
        if self.step_index is None:
            self._init_step_index(timestep)

        if self.drop_cnt is None or self._step_index == 0:
            self._init_ras_config(sample)

        if self._step_index == 0:
            ras_manager.MANAGER.reset_cache()

        latent_dim, height, width = sample.shape[-3:]



        assert ras_manager.MANAGER.sample_ratio > 0.0 and ras_manager.MANAGER.sample_ratio <= 1.0
        if ras_manager.MANAGER.sample_ratio < 1.0 and ras_manager.MANAGER.is_RAS_step:
            model_output.squeeze(0).view(latent_dim, -1)[:, ras_manager.MANAGER.cached_index] = ras_manager.MANAGER.cached_scaled_noise
            model_output = model_output.transpose(0, 1).view(latent_dim, height, width).unsqueeze(0)




        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)
        sigma = self.sigmas[self.step_index]
        print(f"Step {self._step_index}: sigma={sigma}")
        sigma_next = self.sigmas[self.step_index + 1]
        diff = (sigma_next - sigma) * model_output
        prev_sample = sample + diff
        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)




        if ras_manager.MANAGER.std_experiment:
            patch_size = ras_manager.MANAGER.patch_size
            diff_permuted = diff.squeeze(0).permute(1, 2, 0)

            C = diff_permuted.shape[-1]
            num_patches_h = height // patch_size
            num_patches_w = width // patch_size

            current_patched_diff = diff_permuted.view(
                num_patches_h, patch_size, num_patches_w, patch_size, C
            ).permute(0, 2, 1, 3, 4).reshape(-1, patch_size * patch_size * C)

            if self.prevDiff is not None:
                dot_product_per_patch = (current_patched_diff * self.prevDiff).sum(dim=1)
                self.dot_history.append(dot_product_per_patch.detach().cpu())

            self.prevDiff = current_patched_diff.clone().detach()



            if ras_manager.MANAGER.metric == "l2norm":
                metric = torch.norm(diff_permuted, p=2, dim=-1).view(height // ras_manager.MANAGER.patch_size, ras_manager.MANAGER.patch_size, width // ras_manager.MANAGER.patch_size, ras_manager.MANAGER.patch_size).transpose(-2, -3).mean(-1).mean(-1).view(-1)
            else :
                metric = torch.std(diff_permuted, dim=-1).view(height // ras_manager.MANAGER.patch_size, ras_manager.MANAGER.patch_size, width // ras_manager.MANAGER.patch_size, ras_manager.MANAGER.patch_size).transpose(-2, -3).mean(-1).mean(-1).view(-1)



            mean_metric = torch.mean(diff_permuted, dim=-1).view(height // ras_manager.MANAGER.patch_size, ras_manager.MANAGER.patch_size, width // ras_manager.MANAGER.patch_size, ras_manager.MANAGER.patch_size).transpose(-2, -3).mean(-1).mean(-1).view(-1)
            self.std_history.append(metric.clone().detach().cpu())
            self.mean_history.append(mean_metric.clone().detach().cpu())

        if ras_manager.MANAGER.sample_ratio < 1.0 and ras_manager.MANAGER.is_next_RAS_step:
            latent_cached_indices, other_patchified_indices = self.ras_selection(sample, diff, height, width)
            ras_manager.MANAGER.cached_scaled_noise = model_output.squeeze(0).view(latent_dim, -1)[:, latent_cached_indices]
            ras_manager.MANAGER.cached_index = latent_cached_indices
            ras_manager.MANAGER.other_patchified_index = other_patchified_indices
        # upon completion increase step index by one
        self._step_index += 1
        ras_manager.MANAGER.increase_step()
        if ras_manager.MANAGER.current_step >= ras_manager.MANAGER.num_steps:
            if ras_manager.MANAGER.std_experiment:
                self._save_analysis_data()
            ras_manager.MANAGER.reset_cache()

        if not return_dict:
            return (prev_sample,)

        return RASFlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)
