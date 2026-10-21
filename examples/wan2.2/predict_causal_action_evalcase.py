# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import argparse
import os
import sys
from collections import OrderedDict
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf

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
from microworld.models.wan_causal_transformer3d import CausalWanTransformer3DModel
from microworld.models.light_vae_decoder import attach_light_vae
from microworld.pipeline.pipeline_wan_causal import WanCausalPipeline
from microworld.pipeline.pipeline_wan_causal_dmd import WanCausalDMDPipeline
from microworld.utils.lora_utils import merge_lora
from microworld.utils.utils import (
    filter_kwargs,
    get_image_to_video_latent,
    parse_action_list,
    save_videos_grid,
    replace_parameters_by_name,
)
from microworld.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from microworld.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


def parse_args():
    parser = argparse.ArgumentParser(description="Run action eval case inference for Wan2.2 causal TI2V.")
    parser.add_argument("--config_path", type=str, default="config/wan2.2/wan_civitai_5b.yaml")
    parser.add_argument("--model_name", type=str, default="models/Diffusion_Transformer/Wan2.2-TI2V-5B")
    parser.add_argument(
        "--action_checkpoint_path",
        type=str,
        default=None,
        help="Action model checkpoint. If empty/None, run plain TI2V without action control.",
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        default=None,
        help="Causal transformer init weights (CAUSAL_MODEL_PT), used only in the no-action TI2V path.",
    )
    parser.add_argument("--base_lora_path", type=str, default=None)
    parser.add_argument("--base_lora_multiplier", type=float, default=1.0)
    parser.add_argument("--validation_image_start", type=str, default="asset/cliff.jpg")
    parser.add_argument(
        "--prompt",
        type=str,
        default="First-person perspective coastal exploration scene, walking along a cliffside stone path with wooden railings, green bushes lining the walkway, ocean to the left with gentle waves, distant islands visible under a clear sky, \
    realistic head-mounted camera view, smooth forward motion, stable horizon, natural human eye level, high realism, consistent environment, camera motion directly aligned with user actions, immersive and interactive exploration.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    )
    parser.add_argument("--video_height", type=int, default=352)
    parser.add_argument("--video_width", type=int, default=640)
    parser.add_argument("--video_length", type=int, default=81)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--chunk_size", type=int, default=3)
    parser.add_argument("--sink_size", type=int, default=1)
    parser.add_argument("--local_attn_size", type=int, default=21)
    parser.add_argument("--mouse_scale", type=float, default=1.0,
                        help="Multiplier applied to mouse rotation at test time. <1 reduces camera rotation magnitude.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument(
        "--sampler_name",
        type=str,
        default="Flow_Unipc",
        choices=["Flow", "Flow_Unipc", "Flow_DPM++"],
    )
    parser.add_argument(
        "--gpu_memory_mode",
        type=str,
        default="model_full_load",
        choices=["model_full_load", "model_cpu_offload", "sequential_cpu_offload"],
    )
    parser.add_argument("--output_dir", type=str, default="samples/wan-videos-causal-action-evalcase")
    # DMD few-step inference (4 steps, no CFG).
    parser.add_argument("--use_dmd_pipeline", action="store_true", help="Use WanCausalDMDPipeline for few-step inference.")
    parser.add_argument("--denoising_step_list", type=int, nargs="+", default=[1000, 750, 500, 250])
    parser.add_argument("--warp_denoising_step", type=int, default=1, help="1=True, 0=False.")
    parser.add_argument("--num_train_timestep", type=int, default=1000)
    parser.add_argument("--context_noise_level", type=float, default=0.0)
    parser.add_argument("--student_lora_path", type=str, default=None, help="Optional DMD student LoRA merged last.")
    parser.add_argument("--student_lora_multiplier", type=float, default=1.0)
    parser.add_argument(
        "--vae_decode_chunk_frames",
        type=int,
        default=8,
        help="Decode the generated latents in chunks of this many latent frames (1 latent "
             "frame = 4 video frames), keeping only one chunk of pixels on the GPU. This is "
             "what makes 1000+ frame clips decodable; the frame-recurrent VAE cache is kept "
             "across chunks so the result matches a single full decode. 0 = decode in one call.",
    )
    parser.add_argument(
        "--light_vae",
        type=str,
        default="",
        help="Directory with TurboVAED decoder weights (config.json and "
             "diffusion_pytorch_model.safetensors). Empty = original Wan VAE decoder.",
    )
    parser.add_argument(
        "--action_presets",
        type=str,
        nargs="+",
        default=["default"],
        help="Action presets to run, one video per preset. Use 'all' to run every preset. "
             "Available: default, w, s, a, d, mouse_x_pos, mouse_x_neg, mouse_y_pos, "
             "mouse_y_neg, cliff1001.",
    )
    parser.add_argument(
        "--mouse_mag",
        type=float,
        default=DEFAULT_MOUSE_MAG,
        help="Per-frame mouse magnitude (degrees) for the single-axis presets. "
             "Use ~0.3-0.5 for DL3DV-scale models trained after the unit fix.",
    )
    parser.add_argument(
        "--cursor_vis_ref",
        type=float,
        default=0.05,
        help="Visualization-only reference mouse magnitude for the UI cursor. The overlay "
             "is always drawn as if the per-frame mouse base were this value, independent "
             "of --mouse_mag / --mouse_scale, so the cursor never blows up. Does NOT affect "
             "the model.",
    )
    parser.add_argument(
        "--real_action_npz",
        type=str,
        default=None,
        help="Path to an actions.npz (from tools/extract_real_case.py) with keys "
             "'mouse' [T,2]=(pitch,yaw) and 'keyboard' [T,7]. When set, the synthetic "
             "presets are ignored and this real action trajectory is used instead.",
    )
    return parser.parse_args()


def get_weight_dtype(dtype_name: str):
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "fp32":
        return torch.float32
    return torch.bfloat16


def configure_causal_teacher_forcing(transformer3d, chunk_size, independent_first_frame, local_attn_size=None, sink_size=None):
    transformer3d.chunk_size = chunk_size
    transformer3d.independent_first_frame = independent_first_frame
    for block in transformer3d.blocks:
        block.self_attn.chunk_size = chunk_size
        block.self_attn.independent_first_frame = independent_first_frame
        if local_attn_size is not None:
            block.self_attn.local_attn_size = local_attn_size
        if sink_size is not None:
            block.self_attn.sink_size = sink_size


def _unwrap_transformer_state_dict(raw_state_dict):
    if not isinstance(raw_state_dict, (dict, OrderedDict)):
        return raw_state_dict
    for key in ("state_dict", "model", "generator", "module"):
        value = raw_state_dict.get(key, None)
        if isinstance(value, (dict, OrderedDict)):
            return value
    return raw_state_dict


def _normalize_transformer_state_dict_keys(state_dict):
    if not isinstance(state_dict, (dict, OrderedDict)):
        return state_dict
    normalized = OrderedDict()
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "generator.", "model.", "transformer.", "transformer3d."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        normalized[new_key] = value
    return normalized


# Per frame: [w, s, a, d, shift, ctrl, _, mouse_y, mouse_x]; trailing string parsed by parse_action_list.
# Per-frame mouse magnitude for the single-axis presets, in the SAME unit the model
# is fed at train time (degrees, after the dataset's deg->MC->deg handling). DL3DV's
# real per-frame yaw has median ~0.20 / p95 ~0.42 deg, but those peaks are never
# *sustained*; a preset applies a CONSTANT value over the whole clip, so a value near
# the median (~0.15) keeps the accumulated pan in-distribution. Larger sustained values
# (e.g. 0.5 for 20+ frames) push past anything seen in training and the model extrapolates
# into huge rotations. Override via --mouse_mag.
DEFAULT_MOUSE_MAG = 0.15

PRESET_NAMES = ["default", "w", "s", "a", "d", "mouse_x_pos", "mouse_x_neg", "mouse_y_pos", "mouse_y_neg", "cliff1001", "cliff401"]


def _build_fragmented_sequence(pool, total_frames=1000, seg_len=12, start_frame=0, append_jump=True):
    """Build a long fragmented action list by cycling through `pool` (each entry a
    per-frame action string with at most two simultaneous actions), assigning a new
    segment every ~seg_len frames from start_frame up to total_frames. Returns the
    [[end, str], ..., ""] structure consumed by parse_action_list. Set append_jump=False
    to omit the trailing jump-frames "" when concatenating with a leading segment."""
    seq = []
    frame = start_frame
    idx = 0
    while frame < total_frames:
        frame = min(frame + seg_len, total_frames)
        seq.append([frame, pool[idx % len(pool)]])
        idx += 1
    if append_jump:
        seq.append("")
    return seq


def build_action_presets(mouse_mag):
    """Named action presets. Action string order per frame:
    [w, s, a, d, shift, ctrl, _, mouse_y, mouse_x]. A constant action over the whole
    clip is [[80, "..."], ""] (no jump frames). Mouse magnitudes scale with mouse_mag."""
    m = mouse_mag
    # Shared fragmented "seaside-walkway tour" pool (each entry <= two simultaneous actions).
    cliff_pool = [
        "1 0 0 0 0 0 0 0 0",          # forward along path
        f"1 0 0 0 0 0 0 0 {-m}",      # forward + glance left (ocean)
        f"0 0 0 0 0 0 0 0 {-m}",      # pan left to sea
        "0 0 0 0 0 0 0 0 0",          # stop, observe
        f"0 0 0 0 0 0 0 0 {m}",       # pan right back to path
        "1 0 0 0 0 0 0 0 0",          # forward
        f"1 0 0 0 0 0 0 0 {m}",       # forward + glance right (bench/trees)
        f"0 0 0 0 0 0 0 {m} 0",       # look down at stones
        "0 0 1 0 0 0 0 0 0",          # strafe left toward fence
        f"0 0 0 0 0 0 0 {-m} 0",      # look up at sky/horizon
        f"1 0 0 0 0 0 0 0 {-m}",      # forward + glance left
        "0 0 0 0 0 0 0 0 0",          # stop
        f"0 0 0 0 0 0 0 0 {-m}",      # pan left (ocean)
        "1 0 0 0 0 0 0 0 0",          # forward
        "0 0 0 1 0 0 0 0 0",          # strafe right
        f"0 0 0 0 0 0 0 {m} 0",       # look down
        f"1 0 0 0 0 0 0 0 {m}",       # forward + glance right
        "0 1 0 0 0 0 0 0 0",          # back up a little
        f"0 0 0 0 0 0 0 {-m} 0",      # look up
        f"0 0 0 0 0 0 0 0 {m}",       # pan right
    ]
    return {
        # Mixed demo: forward -> back + look right -> strafe right + look left.
        "default": [[20, "1 0 0 0 0 0 0 0 0"], [40, f"0 1 0 0 0 0 0 0 {m}"], [80, f"0 0 0 1 0 0 0 0 {-0.6 * m}"], "30 60"],
        "w": [[80, "1 0 0 0 0 0 0 0 0"], ""],
        "s": [[80, "0 1 0 0 0 0 0 0 0"], ""],
        "a": [[80, "0 0 1 0 0 0 0 0 0"], ""],
        "d": [[80, "0 0 0 1 0 0 0 0 0"], ""],
        "mouse_x_pos": [[80, f"0 0 0 0 0 0 0 0 {m}"], ""],
        "mouse_x_neg": [[80, f"0 0 0 0 0 0 0 0 {-m}"], ""],
        "mouse_y_pos": [[80, f"0 0 0 0 0 0 0 {m} 0"], ""],
        "mouse_y_neg": [[80, f"0 0 0 0 0 0 0 {-m} 0"], ""],
        # 1001-frame case for asset/cliff.jpg: a long fragmented seaside-walkway tour,
        # built by cycling a varied pool (each entry <= two actions). Forward only in
        # short bursts; lots of panning to the ocean, stops, look up/down, side steps.
        "cliff1001": _build_fragmented_sequence(
            cliff_pool,
            total_frames=1000,
            seg_len=12,
        ),
        # Same seaside tour as cliff1001, cut at frame 400 so the clip is 401 frames.
        "cliff401": _build_fragmented_sequence(
            cliff_pool,
            total_frames=400,
            seg_len=12,
        ),
    }

def _pad_or_truncate(t, length):
    if t.shape[0] >= length:
        return t[:length]
    pad = t[-1:].expand(length - t.shape[0], -1)
    return torch.cat([t, pad], dim=0)


def build_pipeline(args, device="cuda", weight_dtype=None):
    """Build the causal TI2V pipeline with the eval-case weight load order.

    Returns ``(pipeline, use_action)``. Factored out of ``main`` so other entry
    points (e.g. the SANA-WM-Bench runner) reuse the exact same load order
    instead of re-deriving it: base -> CAUSAL init -> base LoRA -> action adaLN
    -> DMD student LoRA.
    """
    if weight_dtype is None:
        weight_dtype = get_weight_dtype(args.dtype)
    config = OmegaConf.load(args.config_path)

    use_action = bool(args.action_checkpoint_path)
    if not use_action:
        print("No action checkpoint provided -> running plain TI2V without action control.")

    # Same base for both paths; only the Action class adds action modules.
    # Load order: base -> CAUSAL init (model.pt) -> base LoRA -> action checkpoint.
    model_class = CausalWanActionAdaLNModel if use_action else CausalWanTransformer3DModel
    transformer = model_class.from_pretrained(
        os.path.join(
            args.model_name,
            config["transformer_additional_kwargs"].get("transformer_low_noise_model_subpath", "transformer"),
        ),
        transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
        # low_cpu_mem_usage=False so the action modules (absent from the base checkpoint)
        # are materialized with real storage instead of meta tensors; otherwise the
        # later in-place load_state_dict is a no-op on meta and pipeline.to() crashes.
        low_cpu_mem_usage=False,
        torch_dtype=weight_dtype,
    )
    if use_action:
        configure_causal_teacher_forcing(
            transformer,
            chunk_size=args.chunk_size,
            independent_first_frame=True,
            local_attn_size=args.local_attn_size,
            sink_size=args.sink_size,
        )
    else:
        configure_causal_teacher_forcing(
            transformer,
            chunk_size=args.chunk_size,
            independent_first_frame=True,
        )

    # 1) CAUSAL init weights (CAUSAL_MODEL_PT) -- consistent base for both paths.
    if args.transformer_path:
        print(f"Loading causal transformer init: {args.transformer_path}")
        if args.transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(args.transformer_path)
        else:
            state_dict = torch.load(args.transformer_path, map_location="cpu")
        state_dict = _unwrap_transformer_state_dict(state_dict)
        state_dict = _normalize_transformer_state_dict_keys(state_dict)
        m, u = transformer.load_state_dict(state_dict, strict=False)
        print(f"  transformer missing keys: {len(m)}, unexpected keys: {len(u)}")

    # 2) Base causal TI2V LoRA.
    if args.base_lora_path:
        print(f"Merging base LoRA: {args.base_lora_path}")
        merge_lora(
            SimpleNamespace(transformer=transformer),
            args.base_lora_path,
            args.base_lora_multiplier,
            device="cpu",
            dtype=torch.float32,
            transformer_only=True,
        )

    # 3) Overlay ONLY the action modules (adaLN), keeping the base above intact.
    if use_action:
        print(f"Loading action modules from: {args.action_checkpoint_path}")
        action_model = CausalWanActionAdaLNModel.from_pretrained(
            args.action_checkpoint_path,
            transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
        full_sd = action_model.state_dict()
        action_sd = {
            k: v for k, v in full_sd.items()
            if k.startswith("action_preprocess") or "action_adaLN_modulation" in k
        }
        m, u = transformer.load_state_dict(action_sd, strict=False)
        print(f"  loaded {len(action_sd)} action params; unexpected keys: {len(u)}")
        del action_model, full_sd

    # 4) Optional DMD student LoRA, merged last.
    if args.student_lora_path:
        print(f"Merging DMD student LoRA: {args.student_lora_path}")
        merge_lora(
            SimpleNamespace(transformer=transformer),
            args.student_lora_path,
            args.student_lora_multiplier,
            device="cpu",
            dtype=torch.float32,
            transformer_only=True,
        )

    chosen_autoencoder = {
        "AutoencoderKLWan": AutoencoderKLWan,
        "AutoencoderKLWan3_8": AutoencoderKLWan3_8,
    }[config["vae_kwargs"].get("vae_type", "AutoencoderKLWan")]
    vae = chosen_autoencoder.from_pretrained(
        os.path.join(args.model_name, config["vae_kwargs"].get("vae_subpath", "vae")),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).to(weight_dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.model_name, config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"))
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(args.model_name, config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder")),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    ).eval()

    chosen_scheduler = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }[args.sampler_name]
    if args.sampler_name in ("Flow_Unipc", "Flow_DPM++"):
        config["scheduler_kwargs"]["shift"] = 1
    scheduler = chosen_scheduler(
        **filter_kwargs(chosen_scheduler, OmegaConf.to_container(config["scheduler_kwargs"]))
    )

    pipeline_cls = WanCausalDMDPipeline if args.use_dmd_pipeline else WanCausalPipeline
    pipeline = pipeline_cls(
        transformer=transformer,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
    )
    if args.gpu_memory_mode == "sequential_cpu_offload":
        replace_parameters_by_name(transformer, ["modulation"], device=device)
        if hasattr(transformer, "freqs") and transformer.freqs is not None:
            transformer.freqs = transformer.freqs.to(device=device)
        pipeline.enable_sequential_cpu_offload(device=device)
    elif args.gpu_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    else:
        pipeline.to(device=device)

    if args.light_vae:
        attach_light_vae(pipeline.vae, args.light_vae, device, weight_dtype)

    return pipeline, use_action


def main():
    args = parse_args()
    device = "cuda"
    weight_dtype = get_weight_dtype(args.dtype)
    pipeline, use_action = build_pipeline(args, device=device, weight_dtype=weight_dtype)
    vae = pipeline.vae

    # Resolve which action presets to run; one video is produced per entry.
    action_presets = build_action_presets(args.mouse_mag)
    if use_action and args.real_action_npz:
        # Real trajectory from a training sample: bypass synthetic presets.
        run_presets = [("realcase", "__REAL__")]
    elif use_action:
        if any(name.lower() == "all" for name in args.action_presets):
            run_presets = list(action_presets.items())
        else:
            run_presets = []
            for name in args.action_presets:
                if name not in action_presets:
                    raise ValueError(
                        f"Unknown action preset '{name}'. Available: {list(action_presets.keys())} or 'all'."
                    )
                run_presets.append((name, action_presets[name]))
    else:
        run_presets = [("noaction", None)]

    os.makedirs(args.output_dir, exist_ok=True)
    now = datetime.now().strftime("%Y%m%d-%H%M")
    tag = "action" if use_action else "noaction"
    mode = f"dmd{args.num_inference_steps}step" if args.use_dmd_pipeline else "flow"

    with torch.no_grad():
        video_length = int((args.video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if args.video_length != 1 else 1
        input_video, input_video_mask, _ = get_image_to_video_latent(
            args.validation_image_start,
            None,
            video_length=video_length,
            sample_size=[args.video_height, args.video_width],
        )
        # Keep the (frame-tiled) reference clip on CPU: the pipeline only reads its first
        # frame, and at 1000+ frames a GPU copy would waste a couple of GB. It is reused
        # on CPU for the side-by-side output below.
        input_video = input_video.to(dtype=weight_dtype)
        input_video_mask = torch.zeros_like(input_video[:, :1])
        input_video_mask[:, :, 1:] = 255

        for preset_name, action_list in run_presets:
            # Re-seed per preset so videos differ only by the action input.
            generator = torch.Generator(device=device).manual_seed(args.seed)
            pipeline_kwargs = dict(
                num_frames=video_length,
                negative_prompt=args.negative_prompt,
                height=args.video_height,
                width=args.video_width,
                generator=generator,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                video=input_video[:, :, :video_length],
                mask_video=input_video_mask[:, :, :video_length],
                shift=args.shift,
                chunk_size=args.chunk_size,
            )
            if use_action:
                if action_list == "__REAL__":
                    data = np.load(args.real_action_npz)
                    keyboard_actions = torch.from_numpy(data["keyboard"]).float()
                    mouse_actions = torch.from_numpy(data["mouse"]).float()
                else:
                    keyboard_actions, mouse_actions = parse_action_list(action_list)
                keyboard_actions = _pad_or_truncate(keyboard_actions, video_length)[None].to(device=device, dtype=weight_dtype)
                mouse_actions = _pad_or_truncate(mouse_actions, video_length)[None].to(device=device, dtype=weight_dtype)
                # Scale down (or up) the mouse rotation magnitude at test time.
                mouse_actions = mouse_actions * args.mouse_scale
                # Visualization-only: keep the UI cursor amplitude FIXED at a reference
                # magnitude (cursor_vis_ref) regardless of mouse_mag / mouse_scale, so the
                # overlay does not blow up when those are increased. We divide out the
                # actual magnitude that was baked into the tensor. For synthetic presets
                # the per-frame base is mouse_mag*mouse_scale; for real trajectories there
                # is no single base, so we only divide out mouse_scale (keeping the real
                # degree magnitudes faithful while still ignoring the test-time scaling).
                if action_list == "__REAL__":
                    cursor_vis_scale = 1.0 / args.mouse_scale if args.mouse_scale else 1.0
                else:
                    denom = args.mouse_mag * args.mouse_scale
                    cursor_vis_scale = (args.cursor_vis_ref / denom) if denom else 1.0
                pipeline_kwargs.update(
                    mouse_actions=mouse_actions,
                    keyboard_actions=keyboard_actions,
                    with_ui=True,
                    cursor_vis_scale=cursor_vis_scale,
                )

            if args.use_dmd_pipeline:
                pipeline_kwargs.update(
                    denoising_step_list=args.denoising_step_list,
                    warp_denoising_step=bool(args.warp_denoising_step),
                    num_train_timestep=args.num_train_timestep,
                    context_noise_level=args.context_noise_level,
                )
            pipeline_kwargs.update(vae_decode_chunk_frames=args.vae_decode_chunk_frames)

            print(f"[{preset_name}] Running inference...")
            sample = pipeline(args.prompt, **pipeline_kwargs).videos

            out_path = os.path.join(args.output_dir, f"{now}-evalcase-{tag}-{preset_name}-{mode}.mp4")
            reference = input_video[:, :, :sample.shape[2]].to(sample.device, sample.dtype)
            merged = torch.cat([sample, reference], dim=-1)
            # Long clips make these host tensors several GB each; drop them before the
            # writer builds its own frame buffers.
            del sample, reference
            save_videos_grid(merged, out_path, fps=args.fps)
            del merged
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[{preset_name}] Saved video to {out_path}")


if __name__ == "__main__":
    main()
