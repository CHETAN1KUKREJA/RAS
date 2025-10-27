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
class VayunRASFlowMatchEulerDiscreteSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor


class VayunRASFlowMatchEulerDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
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

        # prediction
        self.max_taylor_order = 4
        self.token_taylor_cache = None
        self.last_step = None
        self.prev_last_step = None
        self.factorials = torch.tensor([math.factorial(i) for i in range(self.max_taylor_order + 1)])

        # code for experimentation evaluation
        if ras_manager.MANAGER.std_experiment:
            self.std_history = None
            self.mean_history = None
            self.prevDiff = None
            self.dot_history = None


    def _init_ras_config(self, latents):
        self.drop_cnt = torch.zeros((latents.shape[-2] // ras_manager.MANAGER.patch_size * latents.shape[-1] // ras_manager.MANAGER.patch_size), device=latents.device) - len(self.sigmas)

        # the prediciton code snippets
        num_patches = (latents.shape[-2] // ras_manager.MANAGER.patch_size) * (latents.shape[-1] // ras_manager.MANAGER.patch_size)
        latent_dim = latents.shape[-3]
        patch_area = ras_manager.MANAGER.patch_size ** 2
        self.token_taylor_cache = torch.zeros(
            num_patches, self.max_taylor_order+1, latent_dim, patch_area,
            device=latents.device, dtype=latents.dtype
        )
        self.last_step = torch.full((num_patches,), -1, device=latents.device, dtype=torch.long)
        self.prev_last_step = torch.full((num_patches,), -1, device=latents.device, dtype=torch.long)
        self.factorials = self.factorials.to(latents.device, dtype=latents.dtype)

        # code for experimentation evaluation        
        if ras_manager.MANAGER.std_experiment:
            self.std_history = []
            self.mean_history = []
            self.prevDiff = None
            self.dot_history = []


    def extract_latents_index_from_patched_latents_index(self, indices, height):
        row_indices = indices // (height // ras_manager.MANAGER.patch_size)
        col_indices = indices % (height // ras_manager.MANAGER.patch_size)
        
        start_indices = row_indices * ras_manager.MANAGER.patch_size * height + col_indices * ras_manager.MANAGER.patch_size
        
        # This creates offsets for a 2x2 patch. Assumes patch_size=2.
        # Offsets should be [0, 1, height, height+1] for a standard row-major layout
        if ras_manager.MANAGER.patch_size != 2:
             logger.warn(f"extract_latents_index_from_patched_latents_index assumes patch_size=2. Results may be incorrect.")
        offsets = torch.tensor([0, 1, height, height + 1], dtype=indices.dtype, device=indices.device)
        flattened_indices = (start_indices.unsqueeze(-1) + offsets).flatten()
        
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
            # if self._step_index is not None and self._step_index % 5 == 0:
            #     highest_deriv = self.token_taylor_cache[updatable_indices, -1]
            #     print(f"[DEBUG]   Updating Derivatives | Highest Order Deriv Max: {highest_deriv.abs().max():.4f}")

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
        # calculate the metric for each patch
        if ras_manager.MANAGER.metric == "std":
            metric = torch.std(diff, dim=-1).view(height // ras_manager.MANAGER.patch_size, ras_manager.MANAGER.patch_size, width // ras_manager.MANAGER.patch_size, ras_manager.MANAGER.patch_size).transpose(-2, -3).mean(-1).mean(-1).view(-1)
        elif ras_manager.MANAGER.metric == "l2norm":
            metric = torch.norm(diff, p=2, dim=-1).view(height // ras_manager.MANAGER.patch_size, ras_manager.MANAGER.patch_size, width // ras_manager.MANAGER.patch_size, ras_manager.MANAGER.patch_size).transpose(-2, -3).mean(-1).mean(-1).view(-1)
        elif ras_manager.MANAGER.metric == "mixture":
            # 1. Calculate both base metrics for each patch
            std_metric = torch.std(diff, dim=-1).view(height // ras_manager.MANAGER.patch_size, ras_manager.MANAGER.patch_size, width // ras_manager.MANAGER.patch_size, ras_manager.MANAGER.patch_size).transpose(-2, -3).mean(-1).mean(-1).view(-1)
            mean_metric = torch.mean(diff, dim=-1).view(height // ras_manager.MANAGER.patch_size, ras_manager.MANAGER.patch_size, width // ras_manager.MANAGER.patch_size, ras_manager.MANAGER.patch_size).transpose(-2, -3).mean(-1).mean(-1).view(-1)

            # 2. Sort both metrics to get ranked indices
            std_indices = torch.sort(std_metric).indices
            mean_indices = torch.sort(mean_metric).indices
            
            num_tokens = std_metric.shape[0]

            low_std_thresh = ras_manager.MANAGER.std_threshold_small
            low_mean_thresh = ras_manager.MANAGER.mean_threshold_small
            high_std_thresh = ras_manager.MANAGER.std_threshold_large
            high_mean_thresh = ras_manager.MANAGER.mean_threshold_large

            low_std_indices = torch.where(std_metric < low_std_thresh)[0]
            low_mean_indices = torch.where(mean_metric < low_mean_thresh)[0]
            high_std_indices = torch.where(std_metric > high_std_thresh)[0]
            high_mean_indices = torch.where(mean_metric > high_mean_thresh)[0]


            low_std_set = set(low_std_indices.tolist())
            low_mean_set = set(low_mean_indices.tolist())
            high_std_set = set(high_std_indices.tolist())
            high_mean_set = set(high_mean_indices.tolist())

            small_set_py = low_std_set.intersection(low_mean_set)
            
            # Set 2: High std AND High mean
            large_set_py = high_std_set.intersection(high_mean_set)

            # 6. Convert to tensors and store them on self (as requested)
            self.small_set = torch.tensor(list(small_set_py), dtype=torch.long, device=std_metric.device)
            self.large_set = torch.tensor(list(large_set_py), dtype=torch.long, device=std_metric.device)

            ras_manager.MANAGER.small_set = self.small_set
            ras_manager.MANAGER.large_set = self.large_set

            print(f"--- RAS Mixture Debug ---")
            print(f"Low/Low Set (small_set) size: {len(ras_manager.MANAGER.small_set)}")
            print(f"Low/Low Set indices: {ras_manager.MANAGER.small_set}")
            print(f"High/High Set (larger_set) size: {len(ras_manager.MANAGER.large_set)}")
            print(f"High/High Set indices: {ras_manager.MANAGER.large_set}")
            print(f"-------------------------")

            union_set = small_set_py.union(large_set_py)

            cached_patchified_indices = torch.tensor(list(union_set), dtype=torch.long, device=std_metric.device)
            print(f"Total cached_patchified_indices (Union): {cached_patchified_indices.shape[0]}")

            # 8. The 'other' indices are all tokens NOT in the union set
            all_indices_set = set(range(num_tokens))
            other_indices_set = all_indices_set.difference(union_set)
            
            other_patchified_indices = torch.tensor(list(other_indices_set), dtype=torch.long, device=std_metric.device)
            # 9. Final processing
            latent_cached_indices = self.extract_latents_index_from_patched_latents_index(cached_patchified_indices, height)
            return latent_cached_indices, other_patchified_indices
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
    ) -> Union[VayunRASFlowMatchEulerDiscreteSchedulerOutput, Tuple]:
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
            num_patches = (sample.shape[-2] // ras_manager.MANAGER.patch_size) * (sample.shape[-1] // ras_manager.MANAGER.patch_size)
            ras_manager.MANAGER.current_active_patchified_index = torch.arange(num_patches, device=sample.device)
            ras_manager.MANAGER.current_active_latent_index= self.extract_latents_index_from_patched_latents_index(ras_manager.MANAGER.current_active_patchified_index, sample.shape[-2])

        latent_dim, height, width = sample.shape[-3:]
        flattened_model_output = model_output.squeeze(0).view(latent_dim, -1)


        assert ras_manager.MANAGER.sample_ratio > 0.0 and ras_manager.MANAGER.sample_ratio <= 1.0
        if ras_manager.MANAGER.sample_ratio < 1.0 and ras_manager.MANAGER.is_RAS_step:
            print(f"[Step {self._step_index}] --- COMBINATION BLOCK (is_RAS_step) ---")
            
            patch_area = ras_manager.MANAGER.patch_size ** 2
            num_patches = self.token_taylor_cache.shape[0] 
            
            combined_features = torch.zeros(
                num_patches, latent_dim, patch_area, 
                device=model_output.device, dtype=model_output.dtype
            )
            
            # Part 1: Active Tokens (Real UNet data)
            active_patch_indices = ras_manager.MANAGER.current_active_patchified_index
            active_latent_indices = ras_manager.MANAGER.current_active_latent_index
            
            if active_patch_indices.numel() > 0:
                original_features_flat = flattened_model_output[:, active_latent_indices]
                original_features_patched = original_features_flat.view(latent_dim, -1, patch_area).permute(1, 0, 2)
                combined_features[active_patch_indices] = original_features_patched
                print(f"[Step {self._step_index}] Using {active_patch_indices.numel()} active tokens")

            # Part 2: Predicted Tokens (Taylor guess)
            if self.large_set is not None and self.large_set.numel() > 0:
                _, predicted_features_all = self._predict_token_output() 
                combined_features[self.large_set] = predicted_features_all[self.large_set].to(combined_features.dtype)
                print(f"[Step {self._step_index}] Predicted {self.large_set.numel()} tokens (large_set)")
            elif self.large_set is None:
                 print(f"[Step {self._step_index}] self.large_set is None, skipping prediction.")
            
            # Part 3: Cached Tokens (Stale data)
            has_direct_cache = ras_manager.MANAGER.cached_scaled_noise is not None
            
            if has_direct_cache:
                if self.small_set is not None and self.small_set.numel() > 0:
                    try:
                        cached_features_patched = ras_manager.MANAGER.cached_scaled_noise.view(latent_dim, -1, patch_area).permute(1, 0, 2)
                        if cached_features_patched.shape[0] == self.small_set.shape[0]:
                            combined_features[self.small_set] = cached_features_patched
                            print(f"[Step {self._step_index}] Directly placed {self.small_set.numel()} tokens (small_set)")
                        else:
                            print(f"[Step {self._step_index}] WARNING: Mismatch in direct cache size. Expected {self.small_set.shape[0]}, got {cached_features_patched.shape[0]}.")
                    except Exception as e:
                        print(f"[Step {self._step_index}] ERROR processing direct cache: {e}")
                elif self.small_set is None:
                    print(f"[Step {self._step_index}] Have direct cache but self.small_set is None.")
            elif self.small_set is not None and self.small_set.numel() > 0:
                print(f"[Step {self._step_index}] self.small_set has {self.small_set.numel()} tokens, but no direct cache was found.")

            # Part 4: Reconstruct
            all_patch_indices = torch.arange(num_patches, device=model_output.device)
            all_latent_indices = self.extract_latents_index_from_patched_latents_index(all_patch_indices, height)
            
            combined_features_flat = combined_features.permute(1, 0, 2).flatten(start_dim=1)
            new_flattened_output = torch.zeros_like(flattened_model_output) 
            new_flattened_output[:, all_latent_indices] = combined_features_flat
            model_output = new_flattened_output.view(1, latent_dim, height, width)




        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)
        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]
        diff = (sigma_next - sigma) * model_output
        prev_sample = sample + diff
        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)



        # code for caching the taylor series
        final_flattened_output = model_output.squeeze(0).view(latent_dim, -1)
        if self._step_index < ras_manager.MANAGER.scheduler_end_step:
            # Get the total number of patches from the taylor cache shape
            num_patches = self.token_taylor_cache.shape[0]
            
            # Create indices for ALL patches
            all_patch_indices = torch.arange(num_patches, device=model_output.device, dtype=torch.long)

            patch_area = ras_manager.MANAGER.patch_size ** 2
            
            # Get latent indices for ALL patches
            latent_indices = self.extract_latents_index_from_patched_latents_index(all_patch_indices, height)
            
            # Get features for ALL patches from the final combined model_output
            features_flat = final_flattened_output[:, latent_indices]
            
            # Reshape features for ALL patches
            all_features = features_flat.view(latent_dim, -1, patch_area).permute(1, 0, 2)
            
            # Update step history for ALL patches
            self.prev_last_step[all_patch_indices] = self.last_step[all_patch_indices]
            self.last_step[all_patch_indices] = self._step_index
            
            # Update derivatives for ALL patches
            self._update_token_derivatives(all_patch_indices, all_features)


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
            print("cache hore hai")
            latent_cached_indices, other_patchified_indices = self.ras_selection(sample, diff, height, width)
            ras_manager.MANAGER.cached_scaled_noise = model_output.squeeze(0).view(latent_dim, -1)[:, self.extract_latents_index_from_patched_latents_index(ras_manager.MANAGER.small_set, height)]
            ras_manager.MANAGER.cached_index = latent_cached_indices
            # this is for placing the prediction
            ras_manager.MANAGER.other_patchified_index = other_patchified_indices
            ras_manager.MANAGER.current_active_latent_index = self.extract_latents_index_from_patched_latents_index(other_patchified_indices, height)
            ras_manager.MANAGER.current_active_patchified_index = other_patchified_indices

        # upon completion increase step index by one
        self._step_index += 1
        ras_manager.MANAGER.increase_step()
        if ras_manager.MANAGER.current_step >= ras_manager.MANAGER.num_steps:
            if ras_manager.MANAGER.std_experiment:
                self._save_analysis_data()
            ras_manager.MANAGER.reset_cache()

        if not return_dict:
            return (prev_sample,)

        return VayunRASFlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)
