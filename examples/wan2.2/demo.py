# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import os
import sys
import time
import math
import json
import base64
import random
import argparse
from io import BytesIO
from threading import Event
from types import SimpleNamespace
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from flask import Flask, jsonify, render_template, send_from_directory
from flask_socketio import SocketIO, emit
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from diffusers.utils.torch_utils import randn_tensor

current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path)),
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))),
]
for project_root in project_roots:
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from microworld.models.wan_vae_3_8 import AutoencoderKLWan3_8
from microworld.models import AutoencoderKLWan, WanT5EncoderModel, AutoTokenizer
from microworld.models.wan_causal_adaln_action import CausalWanActionAdaLNModel
from microworld.pipeline.pipeline_wan_causal import WanCausalPipeline
from microworld.pipeline.pipeline_wan_causal_dmd import WanCausalDMDPipeline
from microworld.utils.lora_utils import merge_lora
from microworld.utils.utils import (
    filter_kwargs,
    parse_action_list,
    get_image_to_video_latent,
    get_video_to_video_latent,
    save_videos_grid,
)
from microworld.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from microworld.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from predict_causal_action_evalcase import build_action_presets

KEYBOARD_LABELS = ["W", "S", "A", "D", "Space", "Shift", "Ctrl"]


def configure_causal_teacher_forcing(transformer3d, chunk_size, independent_first_frame, local_attn_size=-1, sink_size=0):
    transformer3d.chunk_size = chunk_size
    transformer3d.independent_first_frame = independent_first_frame
    for block in transformer3d.blocks:
        block.self_attn.chunk_size = chunk_size
        block.self_attn.independent_first_frame = independent_first_frame
        block.self_attn.local_attn_size = local_attn_size
        block.self_attn.sink_size = sink_size


def tensor_frame_to_data_url(frame_hwc_uint8):
    image = Image.fromarray(frame_hwc_uint8, mode="RGB")
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    payload = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{payload}"


def decoded_chunk_to_serialized_frames(decoded_np):
    # decoded_np: [B, C, F, H, W], value range [0, 1]
    chunk = decoded_np[0].transpose(1, 2, 3, 0)  # [F, H, W, C]
    frames = []
    for idx in range(chunk.shape[0]):
        frame = (np.clip(chunk[idx], 0.0, 1.0) * 255.0).astype(np.uint8)
        frames.append(
            {
                "frame_in_chunk": idx,
                "image": tensor_frame_to_data_url(frame),
            }
        )
    return frames


def clear_vae_decode_cache(vae):
    light = getattr(vae, "light_decoder", None)
    if light is not None:
        light.clear_cache()
        return
    if hasattr(vae, "clear_cache"):
        vae.clear_cache()
        return
    if hasattr(vae, "model") and hasattr(vae.model, "clear_cache"):
        vae.model.clear_cache()


def decode_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    light = getattr(vae, "light_decoder", None)
    if light is not None:
        frames = light.decode(latents.to(vae.dtype))
    elif hasattr(vae, "model") and hasattr(vae.model, "cached_decode"):
        frames = vae.model.cached_decode(latents.to(vae.dtype), vae.scale).clamp_(-1, 1)
    else:
        # Fallback for VAE variants without cached_decode support.
        frames = vae.decode(latents.to(vae.dtype)).sample
    frames = (frames / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
    frames = frames.cpu().float().numpy()
    return frames


def parse_args():
    parser = argparse.ArgumentParser(description="Sprite causal action web demo (streaming)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5002)

    parser.add_argument("--config_path", type=str, default="config/wan2.2/wan_civitai_5b.yaml")
    parser.add_argument("--model_name", type=str, default="ckpts/models/Diffusion_Transformer/Wan2.2-TI2V-5B")
    parser.add_argument("--transformer_path", type=str, default="")
    parser.add_argument("--causal_model_pt", type=str, default="ckpts/causal-forcing-init/model.pt", help="Causal init model.pt, loaded after the base Wan weights.")
    parser.add_argument("--action_checkpoint_path", type=str, default="ckpts/action", help="Directory whose action adaLN modules are copied onto the base.")
    parser.add_argument("--vae_path", type=str, default=None)
    parser.add_argument("--light_vae", type=str, default="", help="TurboVAED decoder directory. Empty keeps the Wan VAE decoder.")
    parser.add_argument("--student_base_lora_path", type=str, default="ckpts/causal-lora/lora_diffusion_pytorch_model.safetensors")
    parser.add_argument("--student_lora_path", type=str, default="ckpts/dmd-lora/lora_diffusion_pytorch_model.safetensors")

    parser.add_argument("--sampler_name", type=str, default="Flow_Unipc", choices=["Flow", "Flow_Unipc", "Flow_DPM++"])
    parser.add_argument("--shift", type=int, default=5)
    parser.add_argument("--weight_dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])

    parser.add_argument("--chunk_size", type=int, default=3)
    parser.add_argument("--independent_first_frame", action="store_true", default=True)
    parser.add_argument("--local_attn_size", type=int, default=21)
    parser.add_argument("--sink_size", type=int, default=1)
    parser.add_argument("--use_dmd_pipeline", action="store_true", default=True)
    parser.add_argument("--denoising_step_list", type=str, default="1000,750,500,250")
    parser.add_argument("--warp_denoising_step", action="store_true", default=True)
    parser.add_argument("--num_train_timestep", type=int, default=1000)
    parser.add_argument("--context_noise_level", type=float, default=0.0)

    parser.add_argument("--video_length", type=int, default=401)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=43)

    parser.add_argument("--save_dir", type=str, default="samples/wan-videos-causal-action-dmd")
    return parser.parse_args()


args = parse_args()
app = Flask(__name__, template_folder="templates")
socketio = SocketIO(app, cors_allowed_origins="*")

generation_active = False
stop_event = Event()
device = torch.device("cuda")

if args.weight_dtype == "bf16":
    weight_dtype = torch.bfloat16
elif args.weight_dtype == "fp16":
    weight_dtype = torch.float16
else:
    weight_dtype = torch.float32

config = OmegaConf.load(args.config_path)

action_list = build_action_presets(0.8)["cliff401"]
base_keyboard_actions, base_mouse_actions = parse_action_list(action_list)


def _pad_or_truncate(t: torch.Tensor, length: int) -> torch.Tensor:
    if t.shape[0] >= length:
        return t[:length]
    pad = t[-1:].expand(length - t.shape[0], -1)
    return torch.cat([t, pad], dim=0)


def build_action_tensors(video_length: int, input_action_list=None):
    src_action_list = input_action_list if input_action_list is not None else action_list
    keyboard_actions, mouse_actions = parse_action_list(src_action_list)
    keyboard = _pad_or_truncate(keyboard_actions, video_length)[None].to(device=device, dtype=weight_dtype)
    mouse = _pad_or_truncate(mouse_actions, video_length)[None].to(device=device, dtype=weight_dtype)
    return keyboard, mouse

def _unwrap_transformer_state_dict(raw_state_dict):
    if not isinstance(raw_state_dict, dict):
        return raw_state_dict
    for key in ("state_dict", "model", "generator", "module"):
        value = raw_state_dict.get(key, None)
        if isinstance(value, dict):
            return value
    return raw_state_dict


def _normalize_transformer_state_dict_keys(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    normalized = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "generator.", "model.", "transformer.", "transformer3d."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        normalized[new_key] = value
    return normalized


base_subpath = config["transformer_additional_kwargs"].get("transformer_low_noise_model_subpath", "transformer")
pretrained_path = os.path.join(args.model_name, base_subpath) if args.causal_model_pt else (
    args.transformer_path if args.transformer_path is not None else os.path.join(args.model_name, base_subpath)
)
transformer = CausalWanActionAdaLNModel.from_pretrained(
    pretrained_path,
    transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
    # Action modules are absent from the base checkpoint. Materialize them so the
    # later in-place load is not a no-op on meta tensors.
    low_cpu_mem_usage=not bool(args.action_checkpoint_path),
    torch_dtype=weight_dtype,
)


configure_causal_teacher_forcing(
    transformer,
    args.chunk_size,
    args.independent_first_frame,
    args.local_attn_size,
    args.sink_size,
)

if args.causal_model_pt:
    print(f"Loading causal transformer init: {args.causal_model_pt}")
    if args.causal_model_pt.endswith("safetensors"):
        from safetensors.torch import load_file
        causal_sd = load_file(args.causal_model_pt)
    else:
        causal_sd = torch.load(args.causal_model_pt, map_location="cpu")
    causal_sd = _normalize_transformer_state_dict_keys(_unwrap_transformer_state_dict(causal_sd))
    missing, unexpected = transformer.load_state_dict(causal_sd, strict=False)
    print(f"  transformer missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

if args.student_base_lora_path:
    print(f"Merging base LoRA: {args.student_base_lora_path}")
    merge_lora(SimpleNamespace(transformer=transformer), args.student_base_lora_path, 1.0, device="cpu", dtype=torch.float32, transformer_only=True)

if args.action_checkpoint_path:
    print(f"Loading action modules from: {args.action_checkpoint_path}")
    action_model = CausalWanActionAdaLNModel.from_pretrained(
        args.action_checkpoint_path,
        transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    action_sd = {
        k: v for k, v in action_model.state_dict().items()
        if k.startswith("action_preprocess") or "action_adaLN_modulation" in k
    }
    _, unexpected = transformer.load_state_dict(action_sd, strict=False)
    print(f"  loaded {len(action_sd)} action params; unexpected keys: {len(unexpected)}")
    del action_model

if args.student_lora_path:
    print(f"Merging DMD student LoRA: {args.student_lora_path}")
    merge_lora(SimpleNamespace(transformer=transformer), args.student_lora_path, 1.0, device="cpu", dtype=torch.float32, transformer_only=True)

Chosen_AutoencoderKL = {
    "AutoencoderKLWan": AutoencoderKLWan,
    "AutoencoderKLWan3_8": AutoencoderKLWan3_8,
}[config["vae_kwargs"].get("vae_type", "AutoencoderKLWan")]

vae = Chosen_AutoencoderKL.from_pretrained(
    os.path.join(args.model_name, config["vae_kwargs"].get("vae_subpath", "vae")),
    additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
).to(weight_dtype)

if args.light_vae:
    from microworld.models.light_vae_decoder import attach_light_vae
    attach_light_vae(vae, args.light_vae, device, weight_dtype)

if args.vae_path:
    if args.vae_path.endswith("safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(args.vae_path)
    else:
        state_dict = torch.load(args.vae_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
    vae.load_state_dict(state_dict, strict=False)

tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(args.model_name, config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"))
)
text_encoder = WanT5EncoderModel.from_pretrained(
    os.path.join(args.model_name, config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder")),
    additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
).eval()

Chosen_Scheduler = {
    "Flow": FlowMatchEulerDiscreteScheduler,
    "Flow_Unipc": FlowUniPCMultistepScheduler,
    "Flow_DPM++": FlowDPMSolverMultistepScheduler,
}[args.sampler_name]
if args.sampler_name in ["Flow_Unipc", "Flow_DPM++"]:
    config["scheduler_kwargs"]["shift"] = 1
scheduler = Chosen_Scheduler(
    **filter_kwargs(Chosen_Scheduler, OmegaConf.to_container(config["scheduler_kwargs"]))
)

PipelineCls = WanCausalDMDPipeline if args.use_dmd_pipeline else WanCausalPipeline
pipeline = PipelineCls(
    transformer=transformer,
    vae=vae,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    scheduler=scheduler,
)
pipeline.to(device=device)


def build_input_video(video_length, sample_size, first_frame_path=None):
    image_path = first_frame_path if first_frame_path is not None else None
    if image_path is not None and str(image_path).strip():
        input_video, input_video_mask, _ = get_image_to_video_latent(
            image_path, None, video_length=video_length, sample_size=sample_size
        )
        input_video_mask = torch.zeros_like(input_video[:, :1])
        input_video_mask[:, :, 1:] = 255
        return input_video, input_video_mask
    return None, None


@torch.no_grad()
def stream_generate(job_id, prompt, seed, guidance_scale, num_inference_steps, requested_video_length, fps, first_frame_path, input_action_list):
    global generation_active
    try:
        generation_active = True
        start_time = time.time()
        sample_size = [args.height, args.width]

        video_length = int((requested_video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if requested_video_length != 1 else 1
        input_video, input_video_mask = build_input_video(video_length, sample_size, first_frame_path=first_frame_path)
        if input_video is not None:
            # Keep the same value range as offline pipeline preprocess: [0, 1] -> [-1, 1].
            input_video = input_video * 2.0 - 1.0
        # Keep causal layout consistent with offline script:
        # i2v -> independent first frame, t2v -> no special first frame chunk.
        run_independent_first_frame = input_video is not None
        configure_causal_teacher_forcing(
            transformer,
            args.chunk_size,
            run_independent_first_frame,
            args.local_attn_size,
            args.sink_size,
        )
        generator = torch.Generator(device=device).manual_seed(seed)
        keyboard_actions, mouse_actions = build_action_tensors(video_length, input_action_list=input_action_list)

        do_cfg = guidance_scale > 1.0
        action_kwargs = {
            "mouse_actions": mouse_actions,
            "keyboard_actions": keyboard_actions,
        }
        negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
        prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_cfg,
            num_videos_per_prompt=1,
            max_sequence_length=512,
            device=device,
            dtype=weight_dtype,
        )

        latent_channels = pipeline.transformer.in_dim
        latents = pipeline.prepare_latents(
            1, latent_channels, video_length, args.height, args.width, weight_dtype, device, generator, None
        )

        reference_latents = None
        if input_video is not None:
            _, reference_latents = pipeline.prepare_mask_latents(
                None,
                input_video[:, :, :1],
                1,
                args.height,
                args.width,
                weight_dtype,
                device,
                generator,
                do_cfg,
                noise_aug_strength=None,
            )

        latent_frames = latents.shape[2]
        latent_height = latents.shape[3]
        latent_width = latents.shape[4]
        frame_seqlen = (latent_height // pipeline.transformer.patch_size[1]) * (latent_width // pipeline.transformer.patch_size[2])
        max_seq_len = latent_frames * frame_seqlen
        num_layers = pipeline.transformer.num_layers
        num_heads = pipeline.transformer.num_heads
        head_dim = pipeline.transformer.dim // num_heads

        kv_caches_cond = pipeline._create_kv_caches(num_layers, max_seq_len, num_heads, head_dim, device, weight_dtype)
        kv_caches_uncond = pipeline._create_kv_caches(num_layers, max_seq_len, num_heads, head_dim, device, weight_dtype)

        start_latent_frame = 0
        if reference_latents is not None:
            ref_frames = reference_latents.shape[2]
            ref_seq_len = ref_frames * frame_seqlen
            ref_timestep = torch.zeros(1, ref_frames, device=device, dtype=torch.long)
            pipeline.transformer(
                x=reference_latents,
                context=prompt_embeds,
                t=ref_timestep,
                seq_len=ref_seq_len,
                kv_caches=kv_caches_cond,
                current_start=0,
                frame_offset=0,
                **action_kwargs,
            )
            if do_cfg:
                pipeline.transformer(
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

        total_chunks = math.ceil((latent_frames - start_latent_frame) / args.chunk_size)
        expected_stream_latent_frames = latent_frames
        socketio.emit(
            "generation_meta",
            {
                "job_id": job_id,
                "requested_video_length": int(requested_video_length),
                "aligned_video_length": int(video_length),
                "fps": int(fps),
                "mode": "i2v" if input_video is not None else "t2v",
                "first_frame_path": first_frame_path if first_frame_path else "",
                "expected_stream_latent_frames": int(expected_stream_latent_frames),
                "expected_chunks": int(total_chunks),
                "keyboard_labels": KEYBOARD_LABELS,
                "action_list": input_action_list if input_action_list is not None else action_list,
            },
        )
        frame_global_index = 0
        generated_latents_for_compare = []
        keyboard_actions_cpu = keyboard_actions[0].detach().to(dtype=torch.float32).cpu()
        mouse_actions_cpu = mouse_actions[0].detach().to(dtype=torch.float32).cpu()

        def emit_latent_frames(latents_to_emit, chunk_index):
            if latents_to_emit is None or latents_to_emit.shape[2] == 0:
                return
            emit_decoded_frames(decode_latents(pipeline.vae, latents_to_emit), chunk_index)

        def emit_decoded_frames(decoded_np, chunk_index):
            nonlocal frame_global_index
            serialized_frames = decoded_chunk_to_serialized_frames(decoded_np)
            for frame in serialized_frames:
                frame["frame_index"] = frame_global_index
                frame["chunk_index"] = chunk_index
                frame["job_id"] = job_id
                key_idx = min(frame_global_index, keyboard_actions_cpu.shape[0] - 1)
                frame["keyboard_state"] = keyboard_actions_cpu[key_idx].tolist()
                frame["mouse_state"] = mouse_actions_cpu[key_idx].tolist()
                socketio.emit("frame_ready", frame)
                frame_global_index += 1

        clear_vae_decode_cache(pipeline.vae)
        light_decoder = getattr(pipeline.vae, "light_decoder", None)
        if reference_latents is not None and reference_latents.shape[2] > 0 and light_decoder is not None:
            # TurboVAED is non-causal and garbles a lone latent frame, so stream the
            # input image itself and only hand its latent over as decode context.
            light_decoder.set_context(reference_latents)
            first_frame_np = ((input_video[:, :, :1] + 1.0) / 2.0).clamp(0, 1).cpu().float().numpy()
            emit_decoded_frames(first_frame_np, -1)
            generated_latents_for_compare.append(reference_latents.detach().to(torch.float32).cpu())
        elif reference_latents is not None and reference_latents.shape[2] > 0:
            # Decode reference frames first so cached_decode state and streamed frames are aligned.
            emit_latent_frames(reference_latents, -1)
            generated_latents_for_compare.append(reference_latents.detach().to(torch.float32).cpu())
        stopped = False
        for c_idx, f_start in enumerate(range(start_latent_frame, latent_frames, args.chunk_size)):
            socketio.sleep(0)
            if stop_event.is_set():
                stopped = True
                break

            f_end = min(f_start + args.chunk_size, latent_frames)
            new_frames = f_end - f_start
            current_start_token = f_start * frame_seqlen
            seq_len_chunk = new_frames * frame_seqlen
            snap_cond = pipeline._snapshot_caches(kv_caches_cond)
            snap_uncond = pipeline._snapshot_caches(kv_caches_uncond)

            if args.use_dmd_pipeline:
                backward_timesteps, backward_sigmas = pipeline._prepare_backward_sim_schedule(
                    num_inference_steps=num_inference_steps,
                    device=device,
                    shift=args.shift,
                    denoising_step_list=[int(x) for x in args.denoising_step_list.split(",")],
                    warp_denoising_step=args.warp_denoising_step,
                    num_train_timestep=args.num_train_timestep,
                )
                chunk_noise = latents[:, :, f_start:f_end].clone()
                latents_chunk = chunk_noise * backward_sigmas[0].to(chunk_noise.dtype)

                for k, t_scalar in enumerate(backward_timesteps):
                    socketio.sleep(0)
                    if stop_event.is_set():
                        stopped = True
                        break
                    pipeline._restore_caches(kv_caches_cond, snap_cond)
                    if do_cfg:
                        pipeline._restore_caches(kv_caches_uncond, snap_uncond)

                    t_chunk = torch.full((1, new_frames), t_scalar.long().item(), device=device, dtype=torch.long)
                    noise_pred_cond = pipeline.transformer(
                        x=latents_chunk,
                        context=prompt_embeds,
                        t=t_chunk,
                        seq_len=seq_len_chunk,
                        kv_caches=kv_caches_cond,
                        current_start=current_start_token,
                        frame_offset=f_start,
                        **action_kwargs,
                    )
                    if do_cfg:
                        noise_pred_uncond = pipeline.transformer(
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
            else:
                timesteps = pipeline._set_timesteps(num_inference_steps, device, args.shift)
                latents_chunk = latents[:, :, f_start:f_end]
                extra_step_kwargs = pipeline.prepare_extra_step_kwargs(generator, eta=0.0)
                for t_scalar in timesteps:
                    socketio.sleep(0)
                    if stop_event.is_set():
                        stopped = True
                        break
                    pipeline._restore_caches(kv_caches_cond, snap_cond)
                    if do_cfg:
                        pipeline._restore_caches(kv_caches_uncond, snap_uncond)
                    t_chunk = torch.full((1, new_frames), t_scalar.long().item(), device=device, dtype=torch.long)
                    noise_pred_cond = pipeline.transformer(
                        x=latents_chunk,
                        context=prompt_embeds,
                        t=t_chunk,
                        seq_len=seq_len_chunk,
                        kv_caches=kv_caches_cond,
                        current_start=current_start_token,
                        frame_offset=f_start,
                        **action_kwargs,
                    )
                    if do_cfg:
                        noise_pred_uncond = pipeline.transformer(
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
                    latents_chunk = pipeline.scheduler.step(noise_pred, t_scalar, latents_chunk, return_dict=False, **extra_step_kwargs)[0]

            if stopped:
                break

            pipeline._restore_caches(kv_caches_cond, snap_cond)
            if do_cfg:
                pipeline._restore_caches(kv_caches_uncond, snap_uncond)

            t_clean = torch.zeros(1, new_frames, device=device, dtype=torch.long)
            clean_context = latents_chunk
            if args.use_dmd_pipeline and args.context_noise_level > 0:
                clean_context = (1.0 - args.context_noise_level) * clean_context + args.context_noise_level * torch.randn_like(clean_context)

            pipeline.transformer(
                x=clean_context,
                context=prompt_embeds,
                t=t_clean,
                seq_len=seq_len_chunk,
                kv_caches=kv_caches_cond,
                current_start=current_start_token,
                frame_offset=f_start,
                **action_kwargs,
            )
            if do_cfg:
                pipeline.transformer(
                    x=clean_context,
                    context=negative_prompt_embeds,
                    t=t_clean,
                    seq_len=seq_len_chunk,
                    kv_caches=kv_caches_uncond,
                    current_start=current_start_token,
                    frame_offset=f_start,
                    **action_kwargs,
                )

            # Keep full-frame streaming and rely on VAE cached_decode state
            # for temporal continuity between chunks.
            emit_latent_frames(latents_chunk, c_idx)
            generated_latents_for_compare.append(latents_chunk.detach().to(torch.float32).cpu())

            socketio.emit(
                "progress",
                {
                    "job_id": job_id,
                    "message": f"chunk {c_idx + 1}/{total_chunks} done",
                    "progress": int((c_idx + 1) * 100 / total_chunks),
                },
            )

        compare_video_path = None
        elapsed = time.time() - start_time
        socketio.emit(
            "generation_complete",
            {
                "job_id": job_id,
                "message": "generation stopped" if stopped else "generation complete",
                "elapsed_sec": round(elapsed, 2),
                "sent_frames": int(frame_global_index),
                "fps": int(fps),
                "full_decode_video_path": compare_video_path if compare_video_path is not None else "",
            },
        )
    except Exception as exc:
        socketio.emit(
            "error",
            {
                "job_id": job_id,
                "message": str(exc),
            },
        )
    finally:
        generation_active = False
        stop_event.set()


@socketio.on("connect")
def on_connect():
    emit("status", {"message": "connected"})


@socketio.on("start_generation")
def on_start_generation(data):
    global generation_active
    if generation_active:
        emit("error", {"message": "generation is already running"})
        return

    prompt = data.get("prompt", "")
    if not prompt:
        emit("error", {"message": "prompt is required"})
        return
    seed = int(data.get("seed", -1))
    if seed < 0:
        seed = random.randint(0, 2**31 - 1)
    guidance_scale = float(data.get("guidance_scale", args.guidance_scale))
    num_inference_steps = int(data.get("num_inference_steps", args.num_inference_steps))
    requested_video_length = int(data.get("video_length", args.video_length))
    fps = int(data.get("fps", args.fps))
    first_frame_path = data.get("first_frame_path", None)
    if isinstance(first_frame_path, str):
        first_frame_path = first_frame_path.strip() or None
    action_list_text = data.get("action_list_text", "")
    parsed_action_list = action_list
    if isinstance(action_list_text, str) and action_list_text.strip():
        try:
            parsed_action_list = json.loads(action_list_text)
            _ = parse_action_list(parsed_action_list)
        except Exception as e:
            emit("error", {"message": f"invalid action_list JSON: {e}"})
            return
    if requested_video_length < 1:
        emit("error", {"message": "video_length must be >= 1"})
        return
    if fps < 1:
        emit("error", {"message": "fps must be >= 1"})
        return
    job_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    generation_active = True
    stop_event.clear()
    socketio.start_background_task(
        stream_generate,
        job_id,
        prompt,
        seed,
        guidance_scale,
        num_inference_steps,
        requested_video_length,
        fps,
        first_frame_path,
        parsed_action_list,
    )
    emit(
        "status",
        {
            "message": "generation started",
            "job_id": job_id,
            "seed": seed,
            "requested_video_length": requested_video_length,
            "fps": fps,
            "mode": "i2v" if first_frame_path else "t2v",
            "first_frame_path": first_frame_path if first_frame_path else "",
        },
    )


@socketio.on("stop_generation")
def on_stop_generation():
    stop_event.set()
    emit("status", {"message": "stop requested"})


@app.route("/")
def index():
    return render_template("demo.html")


@app.route("/asset/<path:filename>")
def asset_file(filename):
    return send_from_directory(os.path.join(project_roots[-1], "asset"), filename)


@app.route("/api/status")
def api_status():
    return jsonify(
        {
            "generation_active": generation_active,
            "sink_size": args.sink_size,
            "local_attn_size": args.local_attn_size,
            "chunk_size": args.chunk_size,
            "use_dmd_pipeline": args.use_dmd_pipeline,
            "default_fps": args.fps,
            "default_video_length": args.video_length,
        }
    )


if __name__ == "__main__":
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Starting Sprite web demo at http://{args.host}:{args.port}")
    socketio.run(app, host=args.host, port=args.port, debug=False, allow_unsafe_werkzeug=True)
