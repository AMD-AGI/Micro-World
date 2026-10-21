# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Measure per-tier inference optimization speedups.

Reuses predict_causal_action_evalcase.build_pipeline so weight loading matches
production inference. Each tier must run in a separate process because all
switches are read from environment variables at import time.

    # A baseline (all off)
    python3 examples/wan2.2/bench_optimizations.py --tag A
    # F all on
    WAN_ADALN_FUSION=script WAN_RMSNORM_FUSION=script WAN_QKV_FUSION=1 \
    WAN_CTX_CACHE=1 WAN_ATTN_AITER=1 python3 examples/wan2.2/bench_optimizations.py --tag F

Generation time is measured per chunk by hooking pipeline._snapshot_caches (once
per chunk before the denoise loop), syncing at chunk boundaries. Steady-state
window defaults to chunk 8 onward: local_attn_size=21 latent frames = 7 chunks
until the KV window fills at chunk 8.

--dump_latents saves generated latents as .pt for cross-tier equivalence checks.
"""
import argparse
import json
import os
import statistics as st
import sys
import time
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict_causal_action_evalcase import (  # noqa: E402
    build_action_presets,
    build_pipeline,
    get_weight_dtype,
    _pad_or_truncate,
)
from microworld.utils.utils import get_image_to_video_latent, parse_action_list  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", type=str, default="run", help="Tier label for output filenames")
    p.add_argument("--config_path", type=str, default="config/wan2.2/wan_civitai_5b.yaml")
    p.add_argument("--model_name", type=str, default="models/Diffusion_Transformer/Wan2.2-TI2V-5B")
    p.add_argument("--action_checkpoint_path", type=str,
                   default="ckpts/action")
    p.add_argument("--base_lora_path", type=str,
                   default="ckpts/causal-lora/lora_diffusion_pytorch_model.safetensors")
    p.add_argument("--student_lora_path", type=str,
                   default="ckpts/dmd-lora/lora_diffusion_pytorch_model.safetensors")
    p.add_argument("--transformer_path", type=str, default="ckpts/causal-forcing-init/model.pt")
    p.add_argument("--base_lora_multiplier", type=float, default=1.0)
    p.add_argument("--student_lora_multiplier", type=float, default=1.0)
    p.add_argument("--validation_image_start", type=str, default="asset/street_night.jpg")
    p.add_argument("--prompt", type=str,
                   default="First-person perspective walking down a lively city street at night. "
                           "Neon signs and bright billboards glow on both sides, cars drive past "
                           "with headlights and taillights streaking slightly.")
    p.add_argument("--negative_prompt", type=str, default="")
    p.add_argument("--video_height", type=int, default=352)
    p.add_argument("--video_width", type=int, default=640)
    p.add_argument("--video_length", type=int, default=201)
    p.add_argument("--num_inference_steps", type=int, default=4)
    p.add_argument("--guidance_scale", type=float, default=1.0,
                   help="DMD 4-step default is 1.0 (no CFG)")
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--chunk_size", type=int, default=3)
    p.add_argument("--sink_size", type=int, default=1)
    p.add_argument("--local_attn_size", type=int, default=21)
    p.add_argument("--mouse_scale", type=float, default=1.0)
    p.add_argument("--mouse_mag", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bf16")
    p.add_argument("--sampler_name", type=str, default="Flow_Unipc")
    p.add_argument("--gpu_memory_mode", type=str, default="model_full_load")
    p.add_argument("--light_vae", type=str, default="",
                   help="TurboVAED weights dir. Empty keeps the original decoder. "
                        "Unused when output_type=latent.")
    p.add_argument("--use_dmd_pipeline", action="store_true", default=True)
    p.add_argument("--denoising_step_list", type=int, nargs="+", default=[1000, 750, 500, 250])
    p.add_argument("--warp_denoising_step", type=int, default=1)
    p.add_argument("--num_train_timestep", type=int, default=1000)
    p.add_argument("--context_noise_level", type=float, default=0.0)
    p.add_argument("--action_preset", type=str, default="w")
    p.add_argument("--runs", type=int, default=2, help="Repeat runs in one process; merge stats")
    p.add_argument("--steady_first", type=int, default=8, help="First chunk index in steady window")
    p.add_argument("--out_dir", type=str, default="samples/bench")
    p.add_argument("--dump_latents", type=str, default=None,
                   help="Save generated latents to this .pt path")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda"
    weight_dtype = get_weight_dtype(args.dtype)

    pipeline, use_action = build_pipeline(args, device=device, weight_dtype=weight_dtype)
    assert use_action, "This benchmark requires --action_checkpoint_path (action path)"
    vae = pipeline.vae

    # Hook chunk boundaries. With guidance_scale>1, _snapshot_caches runs twice per chunk.
    calls_per_chunk = 2 if args.guidance_scale > 1.0 else 1
    marks, state = [], {"n": 0}
    orig_snapshot = pipeline._snapshot_caches

    def snapshot_hook(*a, **kw):
        if state["n"] % calls_per_chunk == 0:
            torch.cuda.synchronize()
            marks.append(time.perf_counter())
        state["n"] += 1
        return orig_snapshot(*a, **kw)

    pipeline._snapshot_caches = snapshot_hook

    with torch.no_grad():
        ratio = vae.config.temporal_compression_ratio
        video_length = int((args.video_length - 1) // ratio * ratio) + 1 if args.video_length != 1 else 1
        input_video, _, _ = get_image_to_video_latent(
            args.validation_image_start, None, video_length=video_length,
            sample_size=[args.video_height, args.video_width])
        input_video = input_video.to(dtype=weight_dtype)
        input_video_mask = torch.zeros_like(input_video[:, :1])
        input_video_mask[:, :, 1:] = 255

        preset = build_action_presets(args.mouse_mag)[args.action_preset]
        keyboard_actions, mouse_actions = parse_action_list(preset)
        keyboard_actions = _pad_or_truncate(keyboard_actions, video_length)[None].to(device, weight_dtype)
        mouse_actions = _pad_or_truncate(mouse_actions, video_length)[None].to(device, weight_dtype) * args.mouse_scale

        all_chunk_ms, latents = [], None
        for run_idx in range(args.runs):
            marks.clear()
            state["n"] = 0
            generator = torch.Generator(device=device).manual_seed(args.seed)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = pipeline(
                args.prompt,
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
                mouse_actions=mouse_actions,
                keyboard_actions=keyboard_actions,
                with_ui=False,
                denoising_step_list=args.denoising_step_list,
                warp_denoising_step=bool(args.warp_denoising_step),
                num_train_timestep=args.num_train_timestep,
                context_noise_level=args.context_noise_level,
                output_type="latent",
            ).videos
            torch.cuda.synchronize()
            total_s = time.perf_counter() - t0
            latents = out
            # Append a final timestamp so the last chunk has a closed interval.
            chunk_ms = [(marks[i + 1] - marks[i]) * 1000.0 for i in range(len(marks) - 1)]
            all_chunk_ms.append(chunk_ms)
            print(f"[{args.tag}] run{run_idx}: {len(chunk_ms)} full chunks, "
                  f"generation-only {total_s:.2f}s")

    steady = [ms for chunk_ms in all_chunk_ms
              for i, ms in enumerate(chunk_ms) if i >= args.steady_first]
    fps = args.chunk_size * 1000.0 / st.median(steady)
    summary = {
        "tag": args.tag,
        "env": {k: os.environ.get(k, "")
                for k in ("WAN_ADALN_FUSION", "WAN_RMSNORM_FUSION", "WAN_QKV_FUSION",
                          "WAN_CTX_CACHE", "WAN_ATTN_AITER")},
        "video_length": video_length,
        "guidance_scale": args.guidance_scale,
        "steady_first": args.steady_first,
        "n_steady": len(steady),
        "gen_p50_ms": st.median(steady),
        "gen_mean_ms": st.mean(steady),
        "gen_std_ms": st.stdev(steady) if len(steady) > 1 else 0.0,
        "gen_fps": fps,
        "per_run_p50_ms": [st.median(c[args.steady_first:]) for c in all_chunk_ms],
        "all_chunk_ms": all_chunk_ms,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"{args.tag}.json")
    json.dump(summary, open(path, "w"), ensure_ascii=False, indent=1)
    print(f"[{args.tag}] steady gen p50={summary['gen_p50_ms']:.2f}ms "
          f"mean={summary['gen_mean_ms']:.2f}±{summary['gen_std_ms']:.2f} "
          f"(n={len(steady)}) gen fps={fps:.3f}  -> {path}")

    if args.dump_latents:
        torch.save(latents.detach().to(torch.float32).cpu(), args.dump_latents)
        print(f"[{args.tag}] latents -> {args.dump_latents}")


if __name__ == "__main__":
    main()
