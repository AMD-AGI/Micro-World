# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import math
from typing import List, Optional, Union

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor

from .pipeline_wan_action_t2w import draw_keys_on_frame, overlay_icon, parse_config
from .pipeline_wan_causal import WanCausalPipeline, WanPipelineOutput


class WanCausalDMDPipeline(WanCausalPipeline):
    """
    Causal DMD inference pipeline.

    This pipeline reuses WanCausalPipeline's prompt/latent/cache utilities and
    switches the per-chunk denoising update from scheduler.step(...) to the
    training-aligned backward-simulation update:
      x0 = x_t - sigma * v
      x_{t_next} = (1 - sigma_next) * x0 + sigma_next * fresh_noise
    """

    def _prepare_backward_sim_schedule(
        self,
        num_inference_steps: int,
        device: torch.device,
        shift: int,
        denoising_step_list: Optional[Union[str, List[int]]] = None,
        warp_denoising_step: bool = True,
        num_train_timestep: int = 1000,
    ):
        if denoising_step_list is not None:
            if isinstance(denoising_step_list, str):
                raw_ts = torch.tensor([int(float(x)) for x in denoising_step_list.split(",")], dtype=torch.long)
            else:
                raw_ts = torch.tensor([int(x) for x in denoising_step_list], dtype=torch.long)

            if warp_denoising_step:
                warp_scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.scheduler.config)
                warp_scheduler.set_timesteps(num_inference_steps=num_train_timestep, device=device)
                warp_table = torch.cat(
                    [warp_scheduler.timesteps.detach().to(device=device).float(), torch.tensor([0.0], device=device)]
                )
                index = (num_train_timestep - raw_ts).clamp(min=0, max=num_train_timestep).to(device=device)
                warped = warp_table[index]
                timesteps = warped.long()
                sigmas = warped.float() / float(num_train_timestep)
            else:
                timesteps = raw_ts.to(device=device)
                sigmas = timesteps.float() / 1000.0
            return timesteps, sigmas

        self.scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
        timesteps = self.scheduler.timesteps
        sigmas = getattr(self.scheduler, "sigmas", None)
        if sigmas is not None:
            sigmas = sigmas.detach().to(device=device).float()
            if sigmas.numel() == timesteps.numel() + 1:
                sigmas = sigmas[:-1]
            elif sigmas.numel() != timesteps.numel():
                sigmas = timesteps.float() / 1000.0
        else:
            sigmas = timesteps.float() / 1000.0
        return timesteps, sigmas

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        num_frames: int = 81,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 352,
        width: int = 640,
        video: Union[torch.FloatTensor] = None,
        mask_video: Union[torch.FloatTensor] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        guidance_scale: float = 6.0,
        num_inference_steps: int = 4,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
        eta: float = 0.0,
        return_dict: bool = False,
        output_type: str = "numpy",
        shift: int = 5,
        chunk_size: int = 4,
        mouse_actions: Optional[torch.Tensor] = None,
        keyboard_actions: Optional[torch.Tensor] = None,
        mouse_scale: Optional[Union[float, List[float]]] = None,
        with_ui: bool = False,
        cursor_vis_scale: float = 1.0,
        denoising_step_list: Optional[Union[str, List[int]]] = None,
        warp_denoising_step: bool = True,
        num_train_timestep: int = 1000,
        context_noise_level: float = 0.0,
        vae_decode_chunk_frames: int = 8,
    ):
        num_videos_per_prompt = 1
        self.check_inputs(prompt, height, width, negative_prompt, prompt_embeds, negative_prompt_embeds)
        self._guidance_scale = guidance_scale

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        weight_dtype = self.transformer.patch_embedding.weight.dtype
        do_classifier_free_guidance = guidance_scale > 1.0
        action_kwargs = {}
        if mouse_actions is not None and keyboard_actions is not None:
            # Scale ONLY the network-bound mouse tensor; the UI overlay below keeps using
            # the raw `mouse_actions`, so the cursor visualization is unaffected.
            net_mouse_actions = mouse_actions.to(device=device, dtype=weight_dtype)
            if mouse_scale is not None:
                _ms = mouse_scale if isinstance(mouse_scale, (list, tuple)) else [mouse_scale]
                _pitch_scale = float(_ms[0])
                _yaw_scale = float(_ms[1]) if len(_ms) > 1 else float(_ms[0])
                if _pitch_scale != 1.0 or _yaw_scale != 1.0:
                    _scale_vec = torch.tensor(
                        [_pitch_scale, _yaw_scale], device=device, dtype=weight_dtype
                    )
                    net_mouse_actions = net_mouse_actions * _scale_vec
            action_kwargs = {
                "mouse_actions": net_mouse_actions,
                "keyboard_actions": keyboard_actions.to(device=device, dtype=weight_dtype),
            }

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=weight_dtype,
        )

        backward_timesteps, backward_sigmas = self._prepare_backward_sim_schedule(
            num_inference_steps=num_inference_steps,
            device=device,
            shift=shift,
            denoising_step_list=denoising_step_list,
            warp_denoising_step=warp_denoising_step,
            num_train_timestep=num_train_timestep,
        )
        self._num_timesteps = len(backward_timesteps)

        if video is not None:
            # Only the first frame is used as clean history below, so preprocess just
            # that one; running the whole (frame-tiled) clip through the image processor
            # would allocate a full-length fp32 pixel tensor for nothing.
            init_video = self.image_processor.preprocess(video[:, :, 0], height=height, width=width)
            init_video = init_video.to(dtype=torch.float32).unsqueeze(2)
        else:
            init_video = None

        latent_channels = self.transformer.in_dim
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            weight_dtype,
            device,
            generator,
            latents,
        )
        reference_latents = None
        if init_video is not None:
            reference_video = init_video[:, :, :1]
            _, reference_latents = self.prepare_mask_latents(
                None,
                reference_video,
                batch_size,
                height,
                width,
                weight_dtype,
                device,
                generator,
                do_classifier_free_guidance,
                noise_aug_strength=None,
            )

        latent_frames = latents.shape[2]
        latent_height = latents.shape[3]
        latent_width = latents.shape[4]

        patch_h = self.transformer.patch_size[1]
        patch_w = self.transformer.patch_size[2]
        frame_seqlen = (latent_height // patch_h) * (latent_width // patch_w)
        max_seq_len = latent_frames * frame_seqlen
        num_layers = self.transformer.num_layers
        num_heads = self.transformer.num_heads
        head_dim = self.transformer.dim // num_heads

        kv_caches_cond = self._create_kv_caches(num_layers, max_seq_len, num_heads, head_dim, device, weight_dtype)
        # The caches scale with the clip length (~80 MB per latent frame for the 5B
        # model at 352x640), so only pay for the uncond branch when CFG is on. 4-step
        # DMD runs with guidance_scale=1.0, i.e. this halves the cache footprint.
        kv_caches_uncond = (
            self._create_kv_caches(num_layers, max_seq_len, num_heads, head_dim, device, weight_dtype)
            if do_classifier_free_guidance
            else None
        )

        generated_latents = reference_latents
        start_latent_frame = 0
        if reference_latents is not None:
            ref_frames = reference_latents.shape[2]
            ref_seq_len = ref_frames * frame_seqlen
            ref_timestep = torch.zeros(batch_size, ref_frames, device=device, dtype=torch.long)
            self.transformer(
                x=reference_latents,
                context=prompt_embeds,
                t=ref_timestep,
                seq_len=ref_seq_len,
                kv_caches=kv_caches_cond,
                current_start=0,
                frame_offset=0,
                **action_kwargs,
            )
            if do_classifier_free_guidance:
                self.transformer(
                    x=reference_latents,
                    context=negative_prompt_embeds,
                    t=ref_timestep,
                    seq_len=ref_seq_len,
                    kv_caches=kv_caches_uncond,
                    current_start=0,
                    frame_offset=0,
                    **action_kwargs,
                )
            start_latent_frame = ref_frames

        num_chunks = math.ceil((latent_frames - start_latent_frame) / chunk_size)
        for c_idx, f_start in enumerate(range(start_latent_frame, latent_frames, chunk_size)):
            f_end = min(f_start + chunk_size, latent_frames)
            new_frames = f_end - f_start
            current_start_token = f_start * frame_seqlen
            seq_len_chunk = new_frames * frame_seqlen

            chunk_noise = latents[:, :, f_start:f_end].clone()
            latents_chunk = chunk_noise * backward_sigmas[0].to(chunk_noise.dtype)

            snap_cond = self._snapshot_caches(kv_caches_cond)
            snap_uncond = self._snapshot_caches(kv_caches_uncond) if do_classifier_free_guidance else None

            for k, t_scalar in enumerate(backward_timesteps):
                self._restore_caches(kv_caches_cond, snap_cond)
                if do_classifier_free_guidance:
                    self._restore_caches(kv_caches_uncond, snap_uncond)

                t_chunk = torch.full(
                    (batch_size, new_frames),
                    t_scalar.long().item(),
                    device=device,
                    dtype=torch.long,
                )

                noise_pred_cond = self.transformer(
                    x=latents_chunk,
                    context=prompt_embeds,
                    t=t_chunk,
                    seq_len=seq_len_chunk,
                    kv_caches=kv_caches_cond,
                    current_start=current_start_token,
                    frame_offset=f_start,
                    **action_kwargs,
                )
                if do_classifier_free_guidance:
                    noise_pred_uncond = self.transformer(
                        x=latents_chunk,
                        context=negative_prompt_embeds,
                        t=t_chunk,
                        seq_len=seq_len_chunk,
                        kv_caches=kv_caches_uncond,
                        current_start=current_start_token,
                        frame_offset=f_start,
                        **action_kwargs,
                    )
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_cond

                sigma_at_step = backward_sigmas[k].to(latents_chunk.dtype)
                x0_pred = latents_chunk - sigma_at_step * noise_pred
                if k < len(backward_timesteps) - 1:
                    sigma_next = backward_sigmas[k + 1].to(latents_chunk.dtype)
                    fresh_noise = randn_tensor(
                        x0_pred.shape,
                        generator=generator,
                        device=device,
                        dtype=latents_chunk.dtype,
                    )
                    latents_chunk = (1.0 - sigma_next) * x0_pred + sigma_next * fresh_noise
                else:
                    latents_chunk = x0_pred

            self._restore_caches(kv_caches_cond, snap_cond)
            if do_classifier_free_guidance:
                self._restore_caches(kv_caches_uncond, snap_uncond)

            x0_context = latents_chunk
            if context_noise_level > 0:
                x0_context = (1.0 - context_noise_level) * x0_context + context_noise_level * torch.randn_like(x0_context)
            t_clean = torch.zeros(batch_size, new_frames, device=device, dtype=torch.long)
            self.transformer(
                x=x0_context,
                context=prompt_embeds,
                t=t_clean,
                seq_len=seq_len_chunk,
                kv_caches=kv_caches_cond,
                current_start=current_start_token,
                frame_offset=f_start,
                **action_kwargs,
            )
            if do_classifier_free_guidance:
                self.transformer(
                    x=x0_context,
                    context=negative_prompt_embeds,
                    t=t_clean,
                    seq_len=seq_len_chunk,
                    kv_caches=kv_caches_uncond,
                    current_start=current_start_token,
                    frame_offset=f_start,
                    **action_kwargs
                )

            generated_latents = torch.cat([generated_latents, latents_chunk], dim=2) if generated_latents is not None else latents_chunk
            print(f"[DMD] Chunk {c_idx + 1}/{num_chunks} done (latent frames {f_start}-{f_end - 1})")

        # The KV caches are dead once the last chunk is generated and dominate GPU
        # memory on long clips, so release them before the VAE decode.
        del kv_caches_cond, kv_caches_uncond
        snap_cond = snap_uncond = latents = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if output_type == "numpy":
            video = self.decode_latents_chunked(generated_latents, vae_decode_chunk_frames)
        elif not output_type == "latent":
            video = self.decode_latents_chunked(generated_latents, vae_decode_chunk_frames)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = generated_latents

        video = video[0]
        if with_ui and mouse_actions is not None and keyboard_actions is not None:
            import cv2

            mouse_icon_path = "./asset/mouse.png"
            mouse_icon = cv2.imread(mouse_icon_path, cv2.IMREAD_UNCHANGED)
            _, _, frame_height, frame_width = video.shape
            # Visualization-only: amplify the drawn cursor motion. Does not affect the
            # generated video. Set for capsaicin/blender-scale deltas (~0.6/frame); at gain
            # 3 the cumulative cursor stays legible without saturating the frame edges.
            # Kept in sync with pipeline_wan_causal.py so DMD/lora validation UIs match.
            cursor_vis_gain = 3.0
            # Do not flip pitch: draw the vertical cursor motion with the same sign as the
            # pitch deltas fed to the model (capsaicin pitch is already MC-signed).
            key_data, mouse_data = parse_config(mouse_actions.float().cpu(), keyboard_actions.cpu(), sample_size=(frame_width, frame_height), cursor_gain=cursor_vis_gain, cursor_y_sign=1.0, cursor_vis_scale=cursor_vis_scale,
                                               cursor_base_scale=1.0, clamp_cursor_to_frame=True)
            mouse_scale = 0.2
            mouse_rotation = -20
            video = video.transpose(1, 2, 3, 0)
            video = (video * 255).astype(np.uint8)
            video = np.ascontiguousarray(video)
            for frame_idx in range(num_frames):
                keys = key_data.get(frame_idx, {"W": False, "A": False, "S": False, "D": False, "Space": False, "Shift": False, "Ctrl": False})
                mouse_position = mouse_data.get(frame_idx, (frame_width // 2, frame_height // 2))
                frame = video[frame_idx]
                draw_keys_on_frame(frame, keys, key_size=(50, 50), spacing=10, bottom_margin=20)
                overlay_icon(frame, mouse_icon, mouse_position, scale=mouse_scale, rotation=mouse_rotation)
            video = video.transpose(3, 0, 1, 2)
            video = (video / 255).astype(np.float32)

        if not return_dict and isinstance(video, np.ndarray):
            video = torch.from_numpy(video)[None]
        elif isinstance(video, torch.Tensor):
            video = video[None]

        return WanPipelineOutput(videos=video)
