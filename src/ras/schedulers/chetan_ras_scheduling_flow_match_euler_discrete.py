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
class ChetanRASFlowMatchEulerDiscreteSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor


class ChetanRASFlowMatchEulerDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
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
                         )
        self.drop_cnt = None
        self.max_taylor_order = 4
        self.token_taylor_cache = None
        self.last_step = None
        self.prev_last_step = None
        self.factorials = torch.tensor([math.factorial(i) for i in range(self.max_taylor_order + 1)])
        if ras_manager.MANAGER.std_experiment:
            self.std_history = None
            self.mean_history = None
            self.prevDiff = None
            self.dot_history = None

    def _init_ras_config(self, latents):
        num_patches = (latents.shape[-2] // ras_manager.MANAGER.patch_size) * (latents.shape[-1] // ras_manager.MANAGER.patch_size)
        self.drop_cnt = torch.zeros((latents.shape[-2] // ras_manager.MANAGER.patch_size * latents.shape[-1] // ras_manager.MANAGER.patch_size), device=latents.device) - len(self.sigmas)
        latent_dim = latents.shape[-3]
        patch_area = ras_manager.MANAGER.patch_size ** 2
        self.token_taylor_cache = torch.zeros(
            num_patches, self.max_taylor_order+1, latent_dim, patch_area,
            device=latents.device, dtype=latents.dtype
        )
        self.last_step = torch.full((num_patches,), -1, device=latents.device, dtype=torch.long)
        self.prev_last_step = torch.full((num_patches,), -1, device=latents.device, dtype=torch.long)
        self.factorials = self.factorials.to(latents.device, dtype=latents.dtype)
        if ras_manager.MANAGER.std_experiment:
            self.std_history, self.mean_history, self.dot_history = [], [], []
            self.prevDiff = None

    def extract_latents_index_from_patched_latents_index(self, indices, height):
        # [DEBUG] Check if this function is scrambling indices
        # if self._step_index is not None and self._step_index % 5 == 0: # Print every 5 steps
            #  print(f"[DEBUG] Index calculation at step {self._step_index}")
            #  print(f"[DEBUG]   Input patch indices (first 5): {indices[:5]}")
        
        row_indices = indices // (height // ras_manager.MANAGER.patch_size)
        col_indices = indices % (height // ras_manager.MANAGER.patch_size)
        
        start_indices = row_indices * ras_manager.MANAGER.patch_size * height + col_indices * ras_manager.MANAGER.patch_size
        
        # This creates offsets for a 2x2 patch. Assumes patch_size=2.
        # Offsets should be [0, 1, height, height+1] for a standard row-major layout
        offsets = torch.tensor([0, 1, height, height + 1], dtype=indices.dtype, device=indices.device)
        flattened_indices = (start_indices.unsqueeze(-1) + offsets).flatten()

        # if self._step_index is not None and self._step_index % 5 == 0:
            #  print(f"[DEBUG]   Output latent indices (first 10): {flattened_indices[:10]}")
        
        return flattened_indices

    def _update_token_derivatives(self, active_patch_indices, active_features):
        if active_patch_indices.numel() == 0: return
        updatable_mask = self.prev_last_step[active_patch_indices] >= 0
        updatable_indices = active_patch_indices[updatable_mask]
        if updatable_indices.numel() > 0:
            old_derivatives = self.token_taylor_cache[updatable_indices].clone()
        self.token_taylor_cache[active_patch_indices, 0] = active_features
        if updatable_indices.numel() > 0:
            for i in range(self.max_taylor_order):
                new_lower_order = self.token_taylor_cache[updatable_indices, i]
                old_lower_order = old_derivatives[:, i]
                self.token_taylor_cache[updatable_indices, i + 1] = new_lower_order - old_lower_order
            # [DEBUG] Check the magnitude of the highest-order derivative
            if self._step_index % 5 == 0:
                highest_deriv = self.token_taylor_cache[updatable_indices, -1]
                # print(f"[DEBUG]   Updating Derivatives | Highest Order Deriv Max: {highest_deriv.abs().max():.4f}")

    def _predict_token_output(self):
        last_steps = self.last_step
        x = self._step_index - last_steps
        derivatives = self.token_taylor_cache
        num_patches = derivatives.shape[0]
        dampening_factor = 1/8
        orders = torch.arange(self.max_taylor_order + 1, device=derivatives.device, dtype=derivatives.dtype)
        x_powers = x.view(-1, 1) ** orders
        coeffs = (x_powers / self.factorials).view(num_patches, -1, 1, 1)
        dampening_weights = (dampening_factor ** orders).view(1, -1, 1, 1)
        predicted_outputs = torch.sum(coeffs * derivatives * dampening_weights, dim=1)
        return last_steps, predicted_outputs

    def ras_selection(self, sample, diff, height, width):
        diff = diff.squeeze(0).permute(1, 2, 0)
        if ras_manager.MANAGER.metric == "std":
            metric = torch.std(diff, dim=-1).view(height // ras_manager.MANAGER.patch_size, ras_manager.MANAGER.patch_size, width // ras_manager.MANAGER.patch_size, ras_manager.MANAGER.patch_size).transpose(-2, -3).mean(-1).mean(-1).view(-1)
        else: # "l2norm"
            metric = torch.norm(diff, p=2, dim=-1).view(height // ras_manager.MANAGER.patch_size, ras_manager.MANAGER.patch_size, width // ras_manager.MANAGER.patch_size, ras_manager.MANAGER.patch_size).transpose(-2, -3).mean(-1).mean(-1).view(-1)
        metric *= torch.exp(ras_manager.MANAGER.starvation_scale * self.drop_cnt)
        current_skip_num = ras_manager.MANAGER.skip_token_num_list[self._step_index + 1]
        indices = torch.sort(metric, dim=0, descending=False).indices
        low_bar = int(current_skip_num * (1 - ras_manager.MANAGER.high_ratio))
        high_bar = int(current_skip_num * ras_manager.MANAGER.high_ratio)
        cached_patchified_indices = torch.cat([indices[:low_bar], indices[-high_bar:]])
        other_patchified_indices = indices[low_bar:-high_bar]
        self.drop_cnt[cached_patchified_indices] += 1
        latent_cached_indices = self.extract_latents_index_from_patched_latents_index(cached_patchified_indices, height)
        return latent_cached_indices, other_patchified_indices, cached_patchified_indices

    def step(self, model_output: torch.FloatTensor, timestep: Union[float, torch.FloatTensor], sample: torch.FloatTensor, **kwargs) -> ChetanRASFlowMatchEulerDiscreteSchedulerOutput:
        if self.step_index is None: self._init_step_index(timestep)
        if self.drop_cnt is None or self._step_index == 0: self._init_ras_config(sample)
        if self._step_index == 0:
            ras_manager.MANAGER.reset_cache()
            num_patches = (sample.shape[-2] // ras_manager.MANAGER.patch_size) * (sample.shape[-1] // ras_manager.MANAGER.patch_size)
            ras_manager.MANAGER.current_active_patchified_index = torch.arange(num_patches, device=sample.device)
            ras_manager.MANAGER.current_active_latent_index= self.extract_latents_index_from_patched_latents_index(ras_manager.MANAGER.current_active_patchified_index, sample.shape[-2])

        latent_dim, height, width = sample.shape[-3:]
        flattened_model_output = model_output.squeeze(0).view(latent_dim, -1)

        # print(f"\n--- [DEBUG] STEP {self._step_index} ---")
        # print(f"[DEBUG] Input MO           | Shape: {model_output.shape} | Min: {model_output.min():.4f} | Max: {model_output.max():.4f} | Mean: {model_output.mean():.4f}")

        if ras_manager.MANAGER.sample_ratio < 1.0 and ras_manager.MANAGER.is_RAS_step:
            # print("[DEBUG] --- PREDICTION/COMBINATION BLOCK ---")
            num_patches, patch_area = self.token_taylor_cache.shape[0], self.token_taylor_cache.shape[-1]

            _, predicted_features_all = self._predict_token_output()
            # print(f"[DEBUG]   Predicted Features | Shape: {predicted_features_all.shape} | Min: {predicted_features_all.min():.4f} | Max: {predicted_features_all.max():.4f} | Mean: {predicted_features_all.mean():.4f} | NaN: {torch.isnan(predicted_features_all).any()}")

            all_patch_indices = torch.arange(num_patches, device=sample.device)
            all_latent_indices = self.extract_latents_index_from_patched_latents_index(all_patch_indices, height)
            original_features_flat = flattened_model_output[:, all_latent_indices]
            original_features_all = original_features_flat.view(latent_dim, num_patches, patch_area).permute(1, 0, 2)
            # print(f"[DEBUG]   Original Features  | Shape: {original_features_all.shape} | Min: {original_features_all.min():.4f} | Max: {original_features_all.max():.4f} | Mean: {original_features_all.mean():.4f}")

            is_cached_mask = torch.zeros(num_patches, dtype=torch.bool, device=sample.device)
            if hasattr(ras_manager.MANAGER, 'cached_patch_indices') and ras_manager.MANAGER.cached_patch_indices.numel() > 0:
                is_cached_mask[ras_manager.MANAGER.cached_patch_indices] = True
            num_cached = is_cached_mask.sum().item()
            # print(f"[DEBUG]   Mask Info        | Cached: {num_cached} / {num_patches} patches ({num_cached/num_patches:.1%})")
            
            is_cached_mask = is_cached_mask.view(-1, 1, 1)
            combined_features = torch.where(is_cached_mask, predicted_features_all, original_features_all)
            # print(f"[DEBUG]   Combined Features  | Shape: {combined_features.shape} | Min: {combined_features.min():.4f} | Max: {combined_features.max():.4f} | Mean: {combined_features.mean():.4f}")

            combined_features_flat = combined_features.permute(1, 0, 2).flatten(start_dim=1)
            new_flattened_output = flattened_model_output.clone()
            new_flattened_output[:, all_latent_indices] = combined_features_flat
            model_output = new_flattened_output.view(1, latent_dim, height, width)
            # print(f"[DEBUG]   Reconstructed MO   | Shape: {model_output.shape} | Min: {model_output.min():.4f} | Max: {model_output.max():.4f} | Mean: {model_output.mean():.4f}")
            # print("[DEBUG] --- END PREDICTION/COMBINATION BLOCK ---")

        sample = sample.to(torch.float32)
        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]
        diff = (sigma_next - sigma) * model_output
        prev_sample = sample + diff
        prev_sample = prev_sample.to(model_output.dtype)

        # print(f"[DEBUG] Final Update       | Diff Min: {diff.min():.4f} | Diff Max: {diff.max():.4f} | Prev Sample Mean: {prev_sample.mean():.4f} | NaN: {torch.isnan(prev_sample).any()}")

        final_flattened_output = model_output.squeeze(0).view(latent_dim, -1)
        if self._step_index < ras_manager.MANAGER.scheduler_end_step:
            active_patch_indices = ras_manager.MANAGER.current_active_patchified_index
            if active_patch_indices.numel() > 0:
                patch_area = ras_manager.MANAGER.patch_size ** 2
                latent_indices = self.extract_latents_index_from_patched_latents_index(active_patch_indices, height)
                features_flat = final_flattened_output[:, latent_indices]
                active_features = features_flat.view(latent_dim, -1, patch_area).permute(1, 0, 2)
                self.prev_last_step[active_patch_indices] = self.last_step[active_patch_indices]
                self.last_step[active_patch_indices] = self._step_index
                self._update_token_derivatives(active_patch_indices, active_features)

        if ras_manager.MANAGER.sample_ratio < 1.0 and ras_manager.MANAGER.is_next_RAS_step:
            latent_cached_indices, other_patchified_indices, cached_patchified_indices = self.ras_selection(sample, diff, height, width)
            ras_manager.MANAGER.cached_index = latent_cached_indices
            ras_manager.MANAGER.cached_patch_indices = cached_patchified_indices
            ras_manager.MANAGER.other_patchified_index = other_patchified_indices
            ras_manager.MANAGER.current_active_latent_index = self.extract_latents_index_from_patched_latents_index(other_patchified_indices, height)
            ras_manager.MANAGER.current_active_patchified_index = other_patchified_indices

        self._step_index += 1
        ras_manager.MANAGER.increase_step()
        if ras_manager.MANAGER.current_step >= ras_manager.MANAGER.num_steps:
            ras_manager.MANAGER.reset_cache()
        
        return ChetanRASFlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)