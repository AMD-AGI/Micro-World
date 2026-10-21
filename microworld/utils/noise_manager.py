# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""
NoiseManager: full three-path error recycling buffer adapted from train_svi.py.

Two buffers, three corruption targets, matching the SVI design exactly:

  noise_error_buffer  ─► corrupt noise           (x_0 / noise end error)
  y_error_buffer      ─► corrupt latents          (x_1 / latent end error)
                       ─► corrupt inpaint_latents  (x_1 / latent end error)

Flow-matching convention used in SVI (straight-path ODE):
  x_0 = source = pure noise ε          (σ = 1 end)
  x_1 = target = clean latent          (σ = 0 end)
  zt  = (1-σ)*x_1 + σ*x_0             (interpolation, σ decreases during inference)
  v   = x_0 - x_1 = ε - latent        (velocity, points from latent toward noise)

Error computation after one training forward pass:

  x_0_pred  = zt + v_pred*(1 - σ)     (self_corr=True,  σ_next=1 → steps to noise end)
  x_0_gt    = zt + v_gt  *(1 - σ)     (= pure noise ε with perfect v)
  noise_error = x_0_pred - x_0_gt     (error in predicting noise)  → noise_error_buffer

  x_1_pred  = zt - σ * v_pred         (self_corr=False, σ_next=0 → steps to latent end)
  x_1_gt    = zt - σ * v_gt           (= clean latent with perfect v)
  y_error   = x_1_pred - x_1_gt       (error in predicting clean latent) → y_error_buffer

  Both errors have the same shape as `latents` (B, C_vae, F, H, W).

Injection in the training step:

  noise_w_error    = noise + sample(noise_error_buffer)
  latents_w_error  = latents + sample(y_error_buffer)
  noisy_latents    = (1-σ)*latents_w_error + σ*noise_w_error       # corrupted zt
  target           = noise_w_error - latents                        # self-corrected velocity (GT x_1 = clean latents)

  inpaint_latents[:, mask_ch:, :y_frame_num, ...]  += sample(y_error_buffer)

Usage in train_noise_lora_i2v.py
─────────────────────────────────
## 1. Construct once before the training loop
from microworld.utils.noise_manager import NoiseManager
noise_manager = NoiseManager(
    num_grids=50,
    buffer_size_per_grid=500,
    error_modulate_factor=0.0,
    buffer_warmup_iter=50,
    noise_prob=0.9,
    y_prob=0.9,
    latent_prob=0.9,
    clean_prob=0.1,
    clean_buffer_update_prob=0.1,
    y_mask_channels=4,       # first 4 channels of inpaint_latents are the mask
    y_error_num_frames=1,    # how many leading latent frames to corrupt in inpaint_latents
)

## 2. Replace the noise / latent / inpaint_latents building block:
noise = torch.randn(latents.size(), ...)

# --- NEW: corrupt all three inputs ---
noise, latents_in, inpaint_latents, use_clean = noise_manager.prepare_inputs(
    noise, latents, timesteps, inpaint_latents
)
noisy_latents = (1.0 - sigmas) * latents_in + sigmas * noise
target        = noise - latents          # self-corrected: GT x_1 (clean latent), not latents_in

## 3. After backward, update buffers:
with torch.no_grad():
    noise_error, y_error = noise_manager.compute_errors(
        noise_pred.detach(), noisy_latents.detach(), latents.detach(), sigmas.detach()
    )
    noise_manager.update_buffers(noise_error, y_error, timesteps, use_clean, accelerator)

## 4. Optionally log:
if global_step % 100 == 0:
    accelerator.log(noise_manager.buffer_stats(), step=global_step)
"""

import random
import torch


class NoiseManager:
    """
    Three-path error recycling buffer matching the train_svi.py design.

    Parameters
    ----------
    num_grids : int
        Number of timestep buckets across [0, 1000].
    buffer_size_per_grid : int
        Max stored error tensors per grid cell (random replacement when full).
    error_modulate_factor : float
        Scale jitter on sampled errors: scale ~ U(1-f, 1+f). 0 = disabled.
    buffer_warmup_iter : int
        First N iters: gather errors from all GPUs via all_gather.
        Afterwards: only local GPU updates the buffer.
    noise_prob : float
        Probability of applying noise_error to the noise tensor.
    y_prob : float
        Probability of applying y_error to inpaint_latents.
    latent_prob : float
        Probability of applying latent_error to the clean latents.
    clean_prob : float
        Probability of forcing a fully clean step (overrides all three above).
    clean_buffer_update_prob : float
        On a clean step, still update the buffer with this probability.
    y_mask_channels : int
        Number of leading channels in inpaint_latents that are the mask
        (not the VAE latent). These are skipped when injecting y_error.
        Default 4 (matches VideoX-Fun: mask shape is (B, 4, F, H, W)).
    y_error_num_frames : int
        How many leading latent frames to corrupt in inpaint_latents.
        Matches train_svi.py `y_error_num`.
    buffer_replacement_strategy : str
        How to replace exsits errors in buffer.
        can be choose in ['random', 'l2_similarity'].
    """

    def __init__(
        self,
        num_grids: int = 50,
        buffer_size_per_grid: int = 100,
        error_modulate_factor: float = 0.0,
        buffer_warmup_iter: int = 50,
        noise_prob: float = 0.9,
        y_prob: float = 0.9,
        latent_prob: float = 0.9,
        clean_prob: float = 0.1,
        clean_buffer_update_prob: float = 0.1,
        y_mask_channels: int = 4,
        y_error_num_frames: int = 1,
        buffer_replacement_strategy: str = "l2_similarity",
    ):
        self.num_grids = num_grids
        self.buffer_size_per_grid = buffer_size_per_grid
        self.error_modulate_factor = error_modulate_factor
        self.buffer_warmup_iter = buffer_warmup_iter
        self.noise_prob = noise_prob
        self.y_prob = y_prob
        self.latent_prob = latent_prob
        self.clean_prob = clean_prob
        self.clean_buffer_update_prob = clean_buffer_update_prob
        self.y_mask_channels = y_mask_channels
        self.y_error_num_frames = y_error_num_frames
        self.buffer_replacement_strategy = buffer_replacement_strategy

        # grid_idx → list[cpu_tensor]
        self.noise_error_buffer: dict[int, list[torch.Tensor]] = {i: [] for i in range(num_grids)}
        self.y_error_buffer:      dict[int, list[torch.Tensor]] = {i: [] for i in range(num_grids)}

        self.iteration_count: int = 0

        # Representative timestep value per grid, built lazily
        self._grid_timesteps: torch.Tensor | None = None

    # ──────────────────────────────────────────────────────────────────
    # Grid mapping
    # ──────────────────────────────────────────────────────────────────

    def _build_grid_timesteps(self, num_train_timesteps: int = 1000):
        step = num_train_timesteps / self.num_grids
        ts = torch.tensor(
            [step * (i + 0.5) for i in range(self.num_grids)], dtype=torch.float32
        )
        self._grid_timesteps = ts

    def _get_grid(self, timestep: torch.Tensor) -> int:
        """Map a (possibly batched) timestep tensor to a single grid index."""
        if self._grid_timesteps is None:
            self._build_grid_timesteps()
        ts_val = timestep.flatten()[0].float().cpu()
        grid_idx = int(torch.argmin((self._grid_timesteps - ts_val).abs()).item())
        return min(grid_idx, self.num_grids - 1)

    # ──────────────────────────────────────────────────────────────────
    # Buffer write
    # ──────────────────────────────────────────────────────────────────
    def _compute_l2_distance_batch(self, new_tensor, stored_tensors):
        """Compute L2 distances between new tensor and all stored tensors efficiently."""
        if not stored_tensors:
            return torch.tensor([])
        
        # Stack all stored tensors for batch computation
        stored_stack = torch.stack(stored_tensors)  # [num_stored, ...]
        new_flat = new_tensor.flatten()
        stored_flat = stored_stack.flatten(start_dim=1)  # [num_stored, flattened_size]
        
        # Compute L2 distances in batch
        distances = torch.norm(stored_flat - new_flat.unsqueeze(0), p=2, dim=1)
        return distances

    def _push(self, buf: list, tensor: torch.Tensor):
        """Add tensor to buf (random replacement when full)."""
        err_cpu = tensor.detach().cpu()
        if len(buf) < self.buffer_size_per_grid:
            buf.append(err_cpu)
        elif self.buffer_replacement_strategy == "l2_similarity":
            distances = self._compute_l2_distance_batch(err_cpu, buf)
            idx = torch.argmin(distances).item()
            buf[idx] = err_cpu
        else:
            buf[random.randint(0, len(buf) - 1)] = err_cpu

    def _update_error(self, error: torch.Tensor, timestep: torch.Tensor, error_buffer: dict):
        self._push(error_buffer[self._get_grid(timestep)], error)

    # ──────────────────────────────────────────────────────────────────
    # Buffer read + modulate
    # ──────────────────────────────────────────────────────────────────

    def _sample(
        self,
        buf: list,
        ref_shape: torch.Size,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """
        Sample one tensor from buf, apply scale modulation, move to device.
        Returns None if buf is empty or shape mismatches.
        """
        if not buf:
            return None
        err = random.choice(buf)
        if err.shape != ref_shape:
            print("Warning, err.shape!=ref_shape, skip error sample.")
            print(err.shape, ref_shape)
            return None
        if self.error_modulate_factor > 0.0:
            lo = 1.0 - self.error_modulate_factor
            hi = 1.0 + self.error_modulate_factor
            err = err * random.uniform(lo, hi)
        return err.to(device=device, dtype=dtype)

    def _has_data(self, buf_dict: dict, grid: int) -> bool:
        return len(buf_dict[grid]) > 0

    # ──────────────────────────────────────────────────────────────────
    # Public API – input preparation
    # ──────────────────────────────────────────────────────────────────

    def prepare_inputs(
        self,
        noise: torch.Tensor,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        inpaint_latents: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, bool]:
        """
        Optionally corrupt noise, latents, and (optionally) inpaint_latents.

        For T2V scripts that have no conditioning latents, pass inpaint_latents=None
        and the y-corruption path is skipped entirely.

        Parameters
        ----------
        noise : (B, C, F, H, W) – clean Gaussian noise
        latents : (B, C, F, H, W) – clean VAE latents (x0)
        timesteps : (B,) – current training timesteps
        inpaint_latents : (B, mask_ch+C, F, H, W) or None – I2V conditioning latents

        Returns
        -------
        noise_out, latents_out, inpaint_latents_out, use_clean
        inpaint_latents_out is None when inpaint_latents is None.
        use_clean : bool – True when a clean step was forced (no corruption)
        """
        self.iteration_count += 1

        # ── decide which errors to apply (batch-level flags) ──────────
        use_clean  = random.random() < self.clean_prob
        add_noise  = (not use_clean) and (random.random() < self.noise_prob)
        add_latent = (not use_clean) and (random.random() < self.latent_prob)
        add_y      = (not use_clean) and (random.random() < self.y_prob)

        noise_out           = noise.clone()
        latents_out         = latents.clone()
        inpaint_latents_out = inpaint_latents.clone() if inpaint_latents is not None else None

        # ── per-sample corruption (each sample may have a different timestep grid) ──
        B = noise.shape[0]
        for i in range(B):
            grid = self._get_grid(timesteps[i])
            has_noise_error = self._has_data(self.noise_error_buffer, grid)
            has_y_error     = self._has_data(self.y_error_buffer, grid)

            # noise corruption
            if add_noise and has_noise_error:
                err = self._sample(
                    self.noise_error_buffer[grid],
                    noise.shape[1:], noise.device, noise.dtype
                )
                if err is not None:
                    noise_out[i] = noise[i] + err

            # latent corruption
            if add_latent and has_y_error:
                err = self._sample(
                    self.y_error_buffer[grid],
                    latents.shape[1:], latents.device, latents.dtype
                )
                if err is not None:
                    latents_out[i] = latents[i] + err

            # y / inpaint_latents corruption
            # Skipped for T2V (inpaint_latents is None).
            # For I2V: inpaint_latents layout is (B, mask_ch + C_vae, F, H, W).
            # Corrupt only the VAE channels ([:, mask_ch:]) for the first
            # y_error_num_frames frames — mirrors SVI's image_emb["y"][:, 4:, :y_error_num, ...]
            if add_y and has_y_error and inpaint_latents_out is not None:
                err = self._sample(
                    self.y_error_buffer[grid],
                    latents.shape[1:], inpaint_latents.device, inpaint_latents.dtype
                )
                if err is not None:
                    max_start = max(0, err.shape[1] - self.y_error_num_frames)  # err: (C, F, H, W)
                    frame_start = random.randint(0, max_start)
                    frame_slice = err[:, frame_start : frame_start + self.y_error_num_frames, ...]
                    mc = self.y_mask_channels
                    inpaint_latents_out[i, mc:, : self.y_error_num_frames, ...] = (
                        inpaint_latents_out[i, mc:, : self.y_error_num_frames, ...]
                        + frame_slice
                    )

        return noise_out, latents_out, inpaint_latents_out, use_clean

    # ──────────────────────────────────────────────────────────────────
    # Public API – error computation
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def compute_errors(
        v_pred: torch.Tensor,
        v_gt: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute noise_error and y_error for buffer update.

        SVI convention:  x_0 = noise (source, σ=1),  x_1 = clean latent (target, σ=0)
          zt  = (1-σ)*x_1 + σ*x_0    (noisy interpolation)
          v   = x_0 - x_1            (velocity = noise - latent, points source→target reversed)

        SVI scheduler step:  prev = zt + v * (σ_next - σ)

        noise_error  — error in predicting x_0 (noise end, self_corr=True → σ_next=1):
          x_0_pred = zt + v_pred * (1 - σ)
          x_0_gt   = zt + v_gt   * (1 - σ)   [= pure noise ε with perfect v]
          noise_error = x_0_pred - x_0_gt = (v_pred - v_gt) * (1 - σ)

        y_error  — error in predicting x_1 (latent end, self_corr=False → σ_next=0):
          x_1_pred = zt + v_pred * (0 - σ) = zt - σ*v_pred
          x_1_gt   = zt - σ*v_gt            [= clean latent with perfect v]
          y_error  = x_1_pred - x_1_gt = (v_pred - v_gt) * (-σ)

        Parameters
        ----------
        v_pred : (B, C, F, H, W) – model velocity output (detached)
        v_gt : (B, C, F, H, W) – model velocity gt (detached)
        sigmas : (B, 1, 1, 1, 1) or broadcastable – σ values

        Returns
        -------
        noise_error : (B, C, F, H, W) – x_0 (noise) prediction error → noise_error_buffer
        y_error     : (B, C, F, H, W) – x_1 (latent) prediction error → y_error_buffer
        """
        # noise_error: error at the x_0 (noise) end, σ_next = 1
        noise_error = (v_pred - v_gt) * (1.0 - sigmas)

        # y_error: error at the x_1 (latent) end, σ_next = 0
        y_error = (v_pred - v_gt) * (-sigmas)

        return noise_error, y_error

    # ──────────────────────────────────────────────────────────────────
    # Public API – buffer update
    # ──────────────────────────────────────────────────────────────────

    def update_buffers(
        self,
        noise_error: torch.Tensor,
        y_error: torch.Tensor,
        timesteps: torch.Tensor,
        use_clean: bool,
        accelerator=None,
    ):
        """
        Push errors into the two buffers.

        Parameters
        ----------
        noise_error : (B, C, F, H, W)
        y_error     : (B, C, F, H, W)
        timesteps   : (B,)
        use_clean   : if True, only update with probability clean_buffer_update_prob
        accelerator : HuggingFace Accelerator (for warmup all_gather); can be None
        """
        bsz = noise_error.shape[0]
        in_warmup = self.iteration_count <= self.buffer_warmup_iter

        if accelerator is not None and in_warmup:
            # All ranks MUST call gather together — do it unconditionally before any
            # early-return, otherwise ranks that skip will deadlock the collective.
            g_noise = accelerator.gather(noise_error)   # (num_gpus*B, C, F, H, W)
            g_y     = accelerator.gather(y_error)
            g_ts    = accelerator.gather(timesteps)     # (num_gpus*B,)

            # Now it is safe to skip the actual buffer write.
            if use_clean and random.random() >= self.clean_buffer_update_prob:
                return

            n = g_noise.shape[0]
            for i in range(n):
                self._update_error(g_noise[i], g_ts[i], self.noise_error_buffer)
                self._update_error(g_y[i], g_ts[i], self.y_error_buffer)
        else:
            # Post-warmup: local GPU only — no collective, safe to skip early.
            if use_clean and random.random() >= self.clean_buffer_update_prob:
                return
            for i in range(bsz):
                self._update_error(noise_error[i], timesteps[i], self.noise_error_buffer)
                self._update_error(y_error[i], timesteps[i], self.y_error_buffer)

    # ──────────────────────────────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────────────────────────────

    def buffer_stats(self) -> dict:
        total_noise = sum(len(v) for v in self.noise_error_buffer.values())
        total_y      = sum(len(v) for v in self.y_error_buffer.values())
        non_empty_noise = sum(1 for v in self.noise_error_buffer.values() if v)
        non_empty_y      = sum(1 for v in self.y_error_buffer.values() if v)
        return {
            "latent_buffer_total": total_noise,
            "y_buffer_total":      total_y,
            "latent_non_empty_grids": non_empty_noise,
            "y_non_empty_grids":      non_empty_y,
            "num_grids":           self.num_grids,
            "iteration_count":     self.iteration_count,
            "in_warmup":           self.iteration_count <= self.buffer_warmup_iter,
        }
