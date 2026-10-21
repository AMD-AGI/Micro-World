# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
from csv import Dialect
import inspect
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional, Union
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.video_processor import VideoProcessor
from diffusers.utils import BaseOutput, logging, replace_example_docstring
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from transformers import T5Tokenizer

from ..models import AutoTokenizer, AutoencoderKLWan, WanT5EncoderModel
from ..utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas
from ..utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .pipeline_wan_ti2v import retrieve_timesteps
from .pipeline_wan_action_t2w import parse_config, draw_keys_on_frame, overlay_icon

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class WanPipelineOutput(BaseOutput):
    r"""
    Output class for CogVideo pipelines.

    Args:
        video (`torch.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
            List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
            denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
            `(batch_size, num_frames, channels, height, width)`.
    """

    videos: torch.Tensor

class WanCausalPipeline(DiffusionPipeline):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: WanT5EncoderModel,
        vae: AutoencoderKLWan,
        transformer,
        transformer_2 = None,
        scheduler: FlowMatchEulerDiscreteScheduler = None,
    ):
        super().__init__()
        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer, 
            transformer_2=transformer_2, scheduler=scheduler
        )
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae.spatial_compression_ratio)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae.spatial_compression_ratio)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae.spatial_compression_ratio, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
        prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=prompt_attention_mask.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return [u[:v] for u, v in zip(prompt_embeds, seq_lens)]

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(
        self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        shape = (
            batch_size,
            num_channels_latents,
            (num_frames - 1) // self.vae.temporal_compression_ratio + 1,
            height // self.vae.spatial_compression_ratio,
            width // self.vae.spatial_compression_ratio,
        )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_mask_latents(
        self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance, noise_aug_strength
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision

        if mask is not None:
            mask = mask.to(device=device, dtype=self.vae.dtype)
            bs = 1
            new_mask = []
            for i in range(0, mask.shape[0], bs):
                mask_bs = mask[i : i + bs]
                mask_bs = self.vae.encode(mask_bs)[0]
                mask_bs = mask_bs.mode()
                new_mask.append(mask_bs)
            mask = torch.cat(new_mask, dim = 0)
            # mask = mask * self.vae.config.scaling_factor

        if masked_image is not None:
            masked_image = masked_image.to(device=device, dtype=self.vae.dtype)
            bs = 1
            new_mask_pixel_values = []
            for i in range(0, masked_image.shape[0], bs):
                mask_pixel_values_bs = masked_image[i : i + bs]
                mask_pixel_values_bs = self.vae.encode(mask_pixel_values_bs)[0]
                mask_pixel_values_bs = mask_pixel_values_bs.mode()
                new_mask_pixel_values.append(mask_pixel_values_bs)
            masked_image_latents = torch.cat(new_mask_pixel_values, dim = 0)
            # masked_image_latents = masked_image_latents * self.vae.config.scaling_factor
        else:
            masked_image_latents = None

        return mask, masked_image_latents
        
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        light = getattr(self.vae, "light_decoder", None)
        if light is not None:
            light.clear_cache()
            frames = light.decode(latents)
        else:
            frames = self.vae.decode(latents.to(self.vae.dtype)).sample
        frames = (frames / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        frames = frames.cpu().float().numpy()
        return frames

    def decode_latents_chunked(self, latents: torch.Tensor, chunk_frames: int = 0) -> np.ndarray:
        """Decode latents in temporal chunks, moving each chunk of pixels to CPU.

        The Wan2.2 VAE decoder is frame-recurrent: it carries a per-conv feature
        cache and already decodes one latent frame at a time. Feeding it
        [0:n), [n:2n), ... with that cache kept alive therefore yields the same
        pixels as a single ``decode`` call, but the decoded video never has to be
        held on the GPU as a whole, which is what blows up for very long clips.
        ``chunk_frames <= 0`` (or a VAE without ``cached_decode``) keeps the
        single-call behaviour.
        """
        light = getattr(self.vae, "light_decoder", None)
        if light is not None:
            if not chunk_frames or chunk_frames <= 0:
                chunk_frames = latents.shape[2]
            latents = latents.to(self.vae.dtype)
            videos = []
            for i in range(latents.shape[0]):
                light.clear_cache()
                chunks = []
                for f_start in range(0, latents.shape[2], chunk_frames):
                    f_end = min(f_start + chunk_frames, latents.shape[2])
                    frames = light.decode(latents[i:i + 1, :, f_start:f_end])
                    frames = (frames / 2 + 0.5).clamp(0, 1).cpu().float().numpy()
                    chunks.append(frames)
                videos.append(chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=2))
            return videos[0] if len(videos) == 1 else np.concatenate(videos, axis=0)

        vae_model = getattr(self.vae, "model", None)
        if not chunk_frames or chunk_frames <= 0 or not hasattr(vae_model, "cached_decode"):
            return self.decode_latents(latents)

        # `cached_decode` bypasses AutoencoderKL.decode, so replay what
        # @apply_forward_hook would have done for the CPU-offload modes.
        hook = getattr(self.vae, "_hf_hook", None)
        if hook is not None and hasattr(hook, "pre_forward"):
            hook.pre_forward(self.vae)

        latents = latents.to(self.vae.dtype)
        videos = []
        for i in range(latents.shape[0]):
            vae_model.clear_cache()
            chunks = []
            for f_start in range(0, latents.shape[2], chunk_frames):
                frames = vae_model.cached_decode(
                    latents[i : i + 1, :, f_start : f_start + chunk_frames], self.vae.scale
                ).clamp_(-1, 1)
                frames = frames.cpu().float()
                chunks.append((frames / 2 + 0.5).clamp_(0, 1).numpy())
                del frames
            vae_model.clear_cache()
            videos.append(chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=2))
            chunks.clear()
        return videos[0] if len(videos) == 1 else np.concatenate(videos, axis=0)

    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

    def _create_kv_caches(self, n_layers, max_len, n_heads, h_dim, device, dtype):
        return [
            {
                "k": torch.zeros(1, max_len, n_heads, h_dim, device=device, dtype=dtype),
                "v": torch.zeros(1, max_len, n_heads, h_dim, device=device, dtype=dtype),
                "global_end_index": torch.tensor(0, dtype=torch.long, device=device),
                "local_end_index": torch.tensor(0, dtype=torch.long, device=device),
            }
            for _ in range(n_layers)
        ]

    def _snapshot_caches(self, caches):
        return [(c["global_end_index"].clone(), c["local_end_index"].clone()) for c in caches]

    def _restore_caches(self, caches, snaps):
        for c, (g, l) in zip(caches, snaps):
            c["global_end_index"].copy_(g)
            c["local_end_index"].copy_(l)

    def _set_timesteps(self, num_inference_steps, device, shift):
        if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, None, mu=1)
        elif isinstance(self.scheduler, FlowUniPCMultistepScheduler):
            self.scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
            timesteps = self.scheduler.timesteps
        elif isinstance(self.scheduler, FlowDPMSolverMultistepScheduler):
            sampling_sigmas = get_sampling_sigmas(num_inference_steps, shift)
            timesteps, _ = retrieve_timesteps(self.scheduler, device=device, sigmas=sampling_sigmas)
        else:
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, None)
        return timesteps

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
        num_inference_steps: int = 30,
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
        with_ui: bool = False,
        cursor_vis_scale: float = 1.0,
        vae_decode_chunk_frames: int = 8,
    ):
        """
        Function invoked when calling the pipeline for causal generation.
        """
        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale

        # 2. Default call parameters
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
            action_kwargs = {
                "mouse_actions": mouse_actions.to(device=device, dtype=weight_dtype),
                "keyboard_actions": keyboard_actions.to(device=device, dtype=weight_dtype),
            }

        # 3. Encode input prompt
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

        # 4. Prepare timesteps
        timesteps = self._set_timesteps(num_inference_steps, device, shift)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latents.
        if video is not None:
            video_length = video.shape[2]
            init_video = self.image_processor.preprocess(rearrange(video, "b c f h w -> (b f) c h w"), height=height, width=width) 
            init_video = init_video.to(dtype=torch.float32)
            init_video = rearrange(init_video, "(b f) c h w -> b c f h w", f=video_length)
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
            # Causal I2V uses only the first input frame as clean history in KV cache.
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

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        patch_h = self.transformer.patch_size[1]
        patch_w = self.transformer.patch_size[2]
        frame_seqlen = (latent_height // patch_h) * (latent_width // patch_w)
        max_seq_len = latent_frames * frame_seqlen
        num_layers = self.transformer.num_layers
        num_heads = self.transformer.num_heads
        head_dim = self.transformer.dim // num_heads
        kv_caches_cond = self._create_kv_caches(num_layers, max_seq_len, num_heads, head_dim, device, weight_dtype)
        # The caches scale with the clip length (~80 MB per latent frame for the 5B
        # model at 352x640), so only pay for the uncond branch when CFG is on.
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

        # 7. Causal denoising loop
        for c_idx, f_start in enumerate(range(start_latent_frame, latent_frames, chunk_size)):
            # Multistep schedulers keep step history, so each independently denoised chunk must start fresh.
            timesteps = self._set_timesteps(num_inference_steps, device, shift)
            self._num_timesteps = len(timesteps)

            f_end = min(f_start + chunk_size, latent_frames)
            new_frames = f_end - f_start
            current_start_token = f_start * frame_seqlen
            seq_len_chunk = new_frames * frame_seqlen
            latents_chunk = latents[:, :, f_start:f_end]

            snap_cond = self._snapshot_caches(kv_caches_cond)
            snap_uncond = self._snapshot_caches(kv_caches_uncond) if do_classifier_free_guidance else None

            for t_scalar in timesteps:
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

                latents_chunk = self.scheduler.step(noise_pred, t_scalar, latents_chunk, return_dict=False, **extra_step_kwargs)[0]

            self._restore_caches(kv_caches_cond, snap_cond)
            if do_classifier_free_guidance:
                self._restore_caches(kv_caches_uncond, snap_uncond)

            t_clean = torch.zeros(batch_size, new_frames, device=device, dtype=torch.long)
            self.transformer(
                x=latents_chunk,
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
                    x=latents_chunk,
                    context=negative_prompt_embeds,
                    t=t_clean,
                    seq_len=seq_len_chunk,
                    kv_caches=kv_caches_uncond,
                    current_start=current_start_token,
                    frame_offset=f_start,
                    **action_kwargs,
                )

            generated_latents = torch.cat([generated_latents, latents_chunk], dim=2) if generated_latents is not None else latents_chunk
            print(f"Chunk {c_idx + 1}/{num_chunks} done (latent frames {f_start}-{f_end - 1})")

        # 8. Decode latents. The KV caches are dead once the last chunk is generated
        # and dominate GPU memory on long clips, so release them before decoding.
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
            c, frame_count, frame_height, frame_width = video.shape
            # Visualization-only: amplify the drawn cursor motion. Does not affect the
            # generated video. Set for capsaicin/blender-scale deltas (~0.6/frame); at gain
            # 3 the cumulative cursor stays legible without saturating the frame edges.
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
