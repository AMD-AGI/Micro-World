# Modifications Copyright(C) [Year of 2025] Advanced Micro Devices, Inc. All rights reserved.

# This file is a modified version of code from VideoXFun
# (https://github.com/aigc-apps/VideoX-Fun),
# which is licensed under the Apache License, Version 2.0.

import inspect
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import numpy as np
import cv2
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from einops import rearrange
from ..models import (AutoencoderKLWan, AutoTokenizer,
                              WanT5EncoderModel, WanTransformer3DModel)
from ..utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                                get_sampling_sigmas)
from ..utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        pass
        ```
"""

def parse_config(mouse_actions, keyboard_actions, sample_size=(640, 352), cursor_gain=1.0, cursor_y_sign=1.0, cursor_vis_scale=1.0,
                 cursor_base_scale=0.4, clamp_cursor_to_frame=False):
    """
    generate key data and mouse data from the configuration
    - actions: input configuration
    - cursor_base_scale: visualization-only base multiplier on the drawn cursor
      displacement, scaled by the frame width relative to the 640px reference. 0.4 is
      the 1.0 default; the 2.0 rollout pipelines use 1.0 for a more legible cursor.
    - clamp_cursor_to_frame: visualization-only. When True the cumulative cursor
      position is clamped to stay inside the frame with a small margin, so long
      rollouts cannot walk the cursor off screen. Off by default (1.0 behaviour).
    - cursor_gain: visualization-only multiplier on the drawn cursor displacement.
      It does NOT affect the model; it only makes small mouse deltas visible in the UI
      overlay (e.g. DL3DV-scale inputs ~0.15/frame would otherwise move ~0.15 px/frame).
    - cursor_y_sign: visualization-only sign for the vertical (pitch / mouse_y) cursor
      motion. DL3DV pitch sign is opposite to the UI's assumed convention, so its
      overlay needs cursor_y_sign=-1 to make the cursor up/down match the actual tilt.
      Does NOT affect the model.
    - cursor_vis_scale: visualization-only multiplier applied to the mouse deltas before
      drawing. Use it to DECOUPLE the overlay amplitude from the model's actual mouse
      magnitude (mouse_mag / mouse_scale): the caller normalizes so the cursor always
      renders at a fixed reference scale (e.g. as if mouse_mag=0.05), no matter how
      large the real input is. Does NOT affect the model.
    - return: key_data and mouse_data

    The drawn cursor is a cumulative position. After the cursor_vis_scale normalization,
    clamp_cursor_to_frame optionally keeps that cumulative position inside the frame as a
    final safety net. Both are visualization-only.
    """
    mouse_actions = mouse_actions[0]
    keyboard_actions = keyboard_actions[0]
    key_data = {}
    mouse_data = {}

    # Keep the cursor on-screen with a small margin so it never blows up.
    margin_x = 0.06 * sample_size[0]
    margin_y = 0.06 * sample_size[1]
    x_min, x_max = margin_x, sample_size[0] - margin_x
    y_min, y_max = margin_y, sample_size[1] - margin_y

    # iterate over each segment of the configuration
    for i in range(len(mouse_actions)):
        w, s, a, d, space, shift, ctrl = keyboard_actions[i]
        mouse_y, mouse_x = mouse_actions[i]

        key_data[i] = {
            "W": bool(w),
            "A": bool(a),
            "S": bool(s),
            "D": bool(d),
            "Space": bool(space),
            "Shift": bool(shift),
            "Ctrl": bool(ctrl),
        }
        # mouse position
        if i == 0:
            mouse_data[i] = (sample_size[0] // 2, sample_size[1] // 2)  # default initial position
        else:
            global_scale_factor = cursor_base_scale * (sample_size[0] / 640)
            mouse_scale_x = global_scale_factor * cursor_gain * cursor_vis_scale
            mouse_scale_y = 4 * global_scale_factor * cursor_gain * cursor_y_sign * cursor_vis_scale
            next_x = mouse_data[i-1][0] + mouse_x * mouse_scale_x  # x coordinate cumulative
            next_y = mouse_data[i-1][1] + mouse_y * mouse_scale_y  # y coordinate cumulative
            if clamp_cursor_to_frame:
                next_x = min(max(next_x, x_min), x_max)
                next_y = min(max(next_y, y_min), y_max)
            mouse_data[i] = (next_x, next_y)

    return key_data, mouse_data


def draw_rounded_rectangle(image, top_left, bottom_right, color, radius=10, alpha=0.5):
    overlay = image.copy()
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)

    cv2.ellipse(overlay, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, -1)
    cv2.ellipse(overlay, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, -1)
    cv2.ellipse(overlay, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, -1)
    cv2.ellipse(overlay, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, -1)

    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

def draw_keys_on_frame(frame, keys, key_size=(80, 50), spacing=20, bottom_margin=30):
    h, w, _ = frame.shape
    horison_shift = 90
    vertical_shift = -20
    horizon_shift_all = 50
    key_positions = {
        "W": (w // 2 - key_size[0] // 2 - horison_shift - horizon_shift_all, h - bottom_margin - key_size[1] * 2 + vertical_shift - 20),
        "A": (w // 2 - key_size[0] * 2 + 5 - horison_shift - horizon_shift_all, h - bottom_margin - key_size[1] + vertical_shift),
        "S": (w // 2 - key_size[0] // 2 - horison_shift - horizon_shift_all, h - bottom_margin - key_size[1] + vertical_shift),
        "D": (w // 2 + key_size[0] - 5 - horison_shift - horizon_shift_all, h - bottom_margin - key_size[1] + vertical_shift),
        "Space": (w // 2 + key_size[0] * 2 + spacing * 2 - horison_shift - horizon_shift_all, h - bottom_margin - key_size[1] + vertical_shift),
        "Shift": (w // 2 + key_size[0] * 3 + spacing * 7 - horison_shift - horizon_shift_all, h - bottom_margin - key_size[1] + vertical_shift),
        "Ctrl": (w // 2 + key_size[0] * 4 + spacing * 12 - horison_shift - horizon_shift_all, h - bottom_margin - key_size[1] + vertical_shift),
    }

    for key, (x, y) in key_positions.items():
        is_pressed = keys.get(key, False)
        top_left = (x, y)
        if key in ["Space", "Shift", "Ctrl"]:
            bottom_right = (x + key_size[0]+40, y + key_size[1])
        else:
            bottom_right = (x + key_size[0], y + key_size[1])

        color = (0, 255, 0) if is_pressed else (200, 200, 200)
        alpha = 0.8 if is_pressed else 0.5

        draw_rounded_rectangle(frame, top_left, bottom_right, color, radius=10, alpha=alpha)

        text_size = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        if key in ["Space", "Shift", "Ctrl"]:
            text_x = x + (key_size[0]+40 - text_size[0]) // 2
        else:
            text_x = x + (key_size[0] - text_size[0]) // 2
        text_y = y + (key_size[1] + text_size[1]) // 2
        cv2.putText(frame, key, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

def overlay_icon(frame, icon, position, scale=1.0, rotation=0):
    x, y = position
    h, w, _ = icon.shape

    # scale icon
    scaled_width = int(w * scale)
    scaled_height = int(h * scale)
    icon_resized = cv2.resize(icon, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)

    # rotate icon
    center = (scaled_width // 2, scaled_height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
    icon_rotated = cv2.warpAffine(icon_resized, rotation_matrix, (scaled_width, scaled_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    h, w, _ = icon_rotated.shape
    frame_h, frame_w, _ = frame.shape

    # calculate drawing region
    top_left_x = max(0, int(x - w // 2))
    top_left_y = max(0, int(y - h // 2))
    bottom_right_x = min(frame_w, int(x + w // 2))
    bottom_right_y = min(frame_h, int(y + h // 2))

    icon_x_start = max(0, int(-x + w // 2))
    icon_y_start = max(0, int(-y + h // 2))
    icon_x_end = icon_x_start + (bottom_right_x - top_left_x)
    icon_y_end = icon_y_start + (bottom_right_y - top_left_y)

    # extract icon region
    icon_region = icon_rotated[icon_y_start:icon_y_end, icon_x_start:icon_x_end]
    # extract frame region
    frame_region = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    # The icon can extend past the frame edge (especially at low resolution where
    # the scaled icon is nearly as tall as the frame). NumPy slicing past array
    # bounds silently returns a smaller region, leaving icon/frame regions with
    # mismatched shapes. Clip both to their common overlap and skip if empty.
    h_ov = min(icon_region.shape[0], frame_region.shape[0])
    w_ov = min(icon_region.shape[1], frame_region.shape[1])
    if h_ov <= 0 or w_ov <= 0:
        return
    icon_region = icon_region[:h_ov, :w_ov]
    frame_region = frame_region[:h_ov, :w_ov]

    alpha = icon_region[:, :, 3:4] / 255.0
    icon_rgb = icon_region[:, :, :3]

    # overlay icon
    blended = (1 - alpha) * frame_region + alpha * icon_rgb
    frame[top_left_y:top_left_y + h_ov, top_left_x:top_left_x + w_ov] = blended


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


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


class WanActionT2WPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-video generation using Wan.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    """

    _optional_components = []
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
    ]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: WanT5EncoderModel,
        vae: AutoencoderKLWan,
        transformer: WanTransformer3DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer, scheduler=scheduler
        )
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae.spacial_compression_ratio)

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

        # duplicate text embeddings for each generation per prompt, using mps friendly method
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
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
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
            height // self.vae.spacial_compression_ratio,
            width // self.vae.spacial_compression_ratio,
        )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma
        return latents
    
    def prepare_control_latents(
        self, control, dtype, device
    ):
        # resize the control to latents shape as we concatenate the control to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision

        if control is not None:
            control = control.to(device=device, dtype=dtype)
            bs = 1
            new_control = []
            for i in range(0, control.shape[0], bs):
                control_bs = control[i : i + bs]
                control_bs = self.vae.encode(control_bs)[0]
                control_bs = control_bs.mode()
                new_control.append(control_bs)
            control = torch.cat(new_control, dim = 0)
        return control

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        frames = self.vae.decode(latents.to(self.vae.dtype)).sample
        frames = (frames / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        frames = frames.cpu().float().numpy()
        return frames

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from diffusers.pipelines.latte.pipeline_latte.LattePipeline.check_inputs
    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt,
        callback_on_step_end_tensor_inputs,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
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

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        mouse_actions = None,
        keyboard_actions = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 720,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "numpy",
        return_dict: bool = False,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        shift: int = 5,
        with_ui: bool = True,
    ) -> Union[WanPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.
        Args:

        Examples:

        Returns:

        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        weight_dtype = self.text_encoder.dtype

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

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
        )
        if do_classifier_free_guidance:
            in_prompt_embeds = negative_prompt_embeds + prompt_embeds
        else:
            in_prompt_embeds = prompt_embeds

        # 4. Prepare timesteps
        if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps, mu=1)
        elif isinstance(self.scheduler, FlowUniPCMultistepScheduler):
            self.scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
            timesteps = self.scheduler.timesteps
        elif isinstance(self.scheduler, FlowDPMSolverMultistepScheduler):
            sampling_sigmas = get_sampling_sigmas(num_inference_steps, shift)
            timesteps, _ = retrieve_timesteps(
                self.scheduler,
                device=device,
                sigmas=sampling_sigmas)
        else:
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latents
        latent_channels = self.vae.latent_channels
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

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        target_shape = (self.vae.latent_channels, (num_frames - 1) // self.vae.temporal_compression_ratio + 1, width // self.vae.spacial_compression_ratio, height // self.vae.spacial_compression_ratio)
        seq_len = math.ceil((target_shape[2] * target_shape[3]) / (self.transformer.config.patch_size[1] * self.transformer.config.patch_size[2]) * target_shape[1]) 
        # 7. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self.transformer.num_inference_steps = num_inference_steps
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                self.transformer.current_steps = i

                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                if hasattr(self.scheduler, "scale_model_input"):
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                keyboard_actions_input = torch.cat([keyboard_actions] * 2) if do_classifier_free_guidance else keyboard_actions
                mouse_actions_input = torch.cat([mouse_actions] * 2) if do_classifier_free_guidance else mouse_actions

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])
                
                # predict noise model_output
                with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=device):
                    noise_pred = self.transformer(
                        x=latent_model_input,
                        context=in_prompt_embeds,
                        mouse_actions=mouse_actions_input,
                        keyboard_actions=keyboard_actions_input,
                        t=timestep,
                        seq_len=seq_len,
                    )

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if output_type == "numpy":
            video = self.decode_latents(latents)
        elif not output_type == "latent":
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents
        video = video[0]
        if with_ui:
            mouse_icon_path = "./asset/mouse.png"
            mouse_icon = cv2.imread(mouse_icon_path, cv2.IMREAD_UNCHANGED)
            key_data, mouse_data = parse_config(mouse_actions.float(), keyboard_actions)
            c, frame_count, frame_height, frame_width = video.shape
            mouse_scale=0.2 
            mouse_rotation=-20
            video = video.transpose(1,2,3,0)
            video = (video * 255).astype(np.uint8)
            video = np.ascontiguousarray(video)
            # ## processing action visuallize
            for frame_idx in range(num_frames):
                keys = key_data.get(frame_idx, {"W": False, "A": False, "S": False, "D": False, "Space": False, "Shift": False, "Ctrl": False})
                mouse_position = mouse_data.get(frame_idx, (frame_width // 2, frame_height // 2))
                frame = video[frame_idx]
                draw_keys_on_frame(frame, keys, key_size=(50, 50), spacing=10, bottom_margin=20)
                overlay_icon(frame, mouse_icon, mouse_position, scale=mouse_scale, rotation=mouse_rotation)
                frame_idx += 1
                print(f"Processing frame {frame_idx}/{frame_count}", end="\r")
            video = video.transpose(3,0,1,2)
            video = (video/255).astype(np.float32)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            # decode_latents can hand back either a numpy array or, for the causal VAE
            # path, a tensor that is already on device.
            if isinstance(video, np.ndarray):
                video = torch.from_numpy(video)[None]
            elif isinstance(video, torch.Tensor):
                video = video[None]

        return WanPipelineOutput(videos=video)