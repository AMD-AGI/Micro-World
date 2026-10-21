# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import argparse
import os
import sys
from datetime import datetime
from multiprocessing import get_context

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path)),
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))),
]
for project_root in project_roots:
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from microworld.data.dataset_game_video import VideoGameDataset
from microworld.models import (
    AutoencoderKLWan,
    AutoTokenizer,
    CLIPModel,
    WanActionAdaLNModel,
    WanActionControlNetModel,
    WanT5EncoderModel,
)
from microworld.models.cache_utils import get_teacache_coefficients
from microworld.pipeline.pipeline_wan_action_i2w import WanActionI2WPipeline
from microworld.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from microworld.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from microworld.utils.lora_utils import merge_lora, unmerge_lora
from microworld.utils.utils import filter_kwargs, replace_parameters_by_name, save_videos_grid, get_image_to_video_latent


def parse_args():
    parser = argparse.ArgumentParser(description="Sprite Action Control Inference (multi-process per GPU)")
    parser.add_argument("--data_dir", type=str, default="/wekafs/yugeng12/datasets/MicroWorld/")
    parser.add_argument("--data_meta", type=str, default="/wekafs/yugeng12/datasets/MicroWorld/video_captions_val.csv")
    parser.add_argument("--data_start", type=int, default=0)
    parser.add_argument("--data_end", type=int, default=None)

    parser.add_argument("--config_path", type=str, default="config/wan2.1/wan_civitai.yaml")
    parser.add_argument("--model_name", type=str, default="models/Diffusion_Transformer/Wan2.1-I2V-14B-480P")
    parser.add_argument("--transformer_path", type=str, default="models/I2W/transformer")
    parser.add_argument("--vae_path", type=str, default=None)
    parser.add_argument("--lora_path", type=str, default="models/I2W/lora_diffusion_pytorch_model.safetensors")
    parser.add_argument("--lora_weight", type=float, default=1.0)
    parser.add_argument("--train_mode", type=str, default="adaln", choices=["adaln", "controlnet"])

    parser.add_argument("--output_prefix", type=str, default=datetime.now().strftime("%y%m%d"))
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--save_subdir_per_gpu", action="store_true", default=False)

    parser.add_argument("--video_length", type=int, default=81)
    parser.add_argument("--sample_height", type=int, default=352)
    parser.add_argument("--sample_width", type=int, default=640)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--shift", type=float, default=3)
    parser.add_argument("--sampler_name", type=str, default="Flow_Unipc", choices=["Flow", "Flow_Unipc", "Flow_DPM++"])
    parser.add_argument("--weight_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])

    parser.add_argument("--gpu_memory_mode", type=str, default="model_full_load", choices=["model_full_load", "model_cpu_offload", "sequential_cpu_offload"])
    parser.add_argument("--enable_teacache", action="store_true", default=False)
    parser.add_argument("--teacache_threshold", type=float, default=0.20)
    parser.add_argument("--num_skip_start_steps", type=int, default=5)
    parser.add_argument("--teacache_offload", action="store_true", default=False)
    parser.add_argument("--cfg_skip_ratio", type=float, default=0.0)

    parser.add_argument("--negative_prompt", type=str, default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")

    parser.add_argument("--gpus", type=str, default="", help="GPU ids, e.g. '0,1,2,3'. Empty means all visible GPUs.")
    return parser.parse_args()


def parse_gpu_ids(gpus_arg):
    if gpus_arg.strip():
        return [int(x.strip()) for x in gpus_arg.split(",") if x.strip()]
    return list(range(torch.cuda.device_count()))


def build_pipeline(args, device, weight_dtype):
    config = OmegaConf.load(args.config_path)
    sample_size = [args.sample_height, args.sample_width]

    if args.train_mode == "controlnet":
        model_class = WanActionControlNetModel
    elif args.train_mode == "adaln":
        model_class = WanActionAdaLNModel
    else:
        raise ValueError(f"Invalid train_mode: {args.train_mode}")

    transformer = model_class.from_pretrained(
        args.transformer_path
        if args.transformer_path is not None
        else os.path.join(args.model_name, config["transformer_additional_kwargs"].get("transformer_subpath", "transformer")),
        transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )

    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(args.model_name, config["vae_kwargs"].get("vae_subpath", "vae")),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).to(weight_dtype)

    if args.vae_path is not None:
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
    clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join(args.model_name, config["image_encoder_kwargs"].get("image_encoder_subpath", "image_encoder"))
    ).to(weight_dtype).eval()

    scheduler_cls = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }[args.sampler_name]
    if args.sampler_name in ("Flow_Unipc", "Flow_DPM++"):
        config["scheduler_kwargs"]["shift"] = 1
    scheduler = scheduler_cls(
        **filter_kwargs(scheduler_cls, OmegaConf.to_container(config["scheduler_kwargs"]))
    )

    pipeline = WanActionI2WPipeline(
        transformer=transformer,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
        clip_image_encoder=clip_image_encoder,
    )

    if args.gpu_memory_mode == "sequential_cpu_offload":
        replace_parameters_by_name(transformer, ["modulation"], device=device)
        transformer.freqs = transformer.freqs.to(device=device)
        pipeline.enable_sequential_cpu_offload(device=device)
    elif args.gpu_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    else:
        pipeline.to(device=device)

    coefficients = get_teacache_coefficients(args.model_name) if args.enable_teacache else None
    if coefficients is not None:
        pipeline.transformer.enable_teacache(
            coefficients,
            args.num_inference_steps,
            args.teacache_threshold,
            num_skip_start_steps=args.num_skip_start_steps,
            offload=args.teacache_offload,
        )
    if args.cfg_skip_ratio is not None and args.cfg_skip_ratio > 0:
        pipeline.transformer.enable_cfg_skip(args.cfg_skip_ratio, args.num_inference_steps)

    if args.lora_path is not None:
        pipeline = merge_lora(pipeline, args.lora_path, args.lora_weight, device=device)

    return pipeline, vae, sample_size


def infer_worker(proc_rank, gpu_id, args, all_indices, save_path):
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    dtype = torch.bfloat16 if args.weight_dtype == "bfloat16" else torch.float16

    worker_indices = all_indices[proc_rank::len(args._gpu_ids)]
    if len(worker_indices) == 0:
        print(f"[GPU {gpu_id}] no samples assigned, skip.")
        return

    val_dataset = VideoGameDataset(
        ann_path=args.data_meta,
        data_root=args.data_dir,
        video_sample_size=[args.sample_height, args.sample_width],
        video_sample_n_frames=args.video_length,
        use_action=True,
        text_drop_ratio=0.0,
    )

    pipeline, vae, sample_size = build_pipeline(args, device=device, weight_dtype=dtype)

    video_length = (
        int((args.video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1
        if args.video_length != 1
        else 1
    )

    if args.save_subdir_per_gpu:
        worker_save_path = os.path.join(save_path, f"gpu_{gpu_id}")
    else:
        worker_save_path = save_path
    os.makedirs(worker_save_path, exist_ok=True)

    gen = torch.Generator(device=device)
    iterator = tqdm(worker_indices, desc=f"GPU{gpu_id}", position=proc_rank)
    with torch.no_grad():
        for data_idx in iterator:
            data = val_dataset[data_idx]
            first_frame = (data["pixel_values"][0] + 1) / 2
            input_video = torch.zeros([1, 3, video_length, args.sample_height, args.sample_width])
            input_video[0, :, 0] = first_frame
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, 1:] = 255
            clip_image = (first_frame*255).to(torch.uint8).permute(1,2,0)
            clip_image = Image.fromarray(clip_image.cpu().numpy())
            gen.manual_seed(args.seed + data_idx)

            sample = pipeline(
                data["text"],
                mouse_actions=data["mouse_actions"][None].to(device=device, dtype=dtype),
                keyboard_actions=data["keyboard_actions"][None].to(device=device, dtype=dtype),
                num_frames=video_length,
                negative_prompt=args.negative_prompt,
                height=sample_size[0],
                width=sample_size[1],
                video=input_video,
                mask_video=input_video_mask,
                clip_image=clip_image,
                generator=gen,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                shift=args.shift,
                with_ui=False,
            ).videos

            prefix = f"{data_idx:06d}_{data['name']}"
            if video_length == 1:
                image_path = os.path.join(worker_save_path, f"{prefix}.png")
                image = sample[0, :, 0].transpose(0, 1).transpose(1, 2)
                image = (image * 255).cpu().numpy().astype(np.uint8)
                Image.fromarray(image).save(image_path)
            else:
                video_path = os.path.join(worker_save_path, f"{prefix}.mp4")
                save_videos_grid(sample, video_path, fps=args.fps)

    if args.lora_path is not None:
        pipeline = unmerge_lora(pipeline, args.lora_path, args.lora_weight, device=device)
    del pipeline
    torch.cuda.empty_cache()
    print(f"[GPU {gpu_id}] finished {len(worker_indices)} samples.")


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    args._gpu_ids = parse_gpu_ids(args.gpus)
    if len(args._gpu_ids) == 0:
        raise ValueError("No GPU selected. Set --gpus or make GPUs visible.")

    if args.save_path:
        save_path = args.save_path
    else:
        save_path = f"samples/wan-videos-I2W/infer_{args.output_prefix}"
    os.makedirs(save_path, exist_ok=True)

    val_dataset_len = len(
        VideoGameDataset(
            ann_path=args.data_meta,
            data_root=args.data_dir,
            video_sample_size=[args.sample_height, args.sample_width],
            video_sample_n_frames=args.video_length,
            use_action=True,
            text_drop_ratio=0.0,
        )
    )
    data_start = max(0, args.data_start)
    data_end = val_dataset_len if args.data_end is None else min(args.data_end, val_dataset_len)
    if data_end <= data_start:
        raise ValueError(f"Invalid range: [{data_start}, {data_end})")

    all_indices = list(range(data_start, data_end))

    print("\n" + "=" * 60)
    print("Inference Configuration")
    print("=" * 60)
    print(f"Data meta:          {args.data_meta}")
    print(f"Data dir:           {args.data_dir}")
    print(f"Data range:         [{data_start}, {data_end}) total={len(all_indices)}")
    print(f"GPUs:               {args._gpu_ids}")
    print(f"Save path:          {save_path}")
    print(f"Train mode:         {args.train_mode}")
    print(f"Model name:         {args.model_name}")
    print(f"Transformer path:   {args.transformer_path}")
    print(f"LoRA path:          {args.lora_path}")
    print(f"Video length:       {args.video_length}")
    print(f"Resolution:         {args.sample_height}x{args.sample_width}")
    print(f"Steps / CFG:        {args.num_inference_steps} / {args.guidance_scale}")
    print("=" * 60 + "\n")

    mp_ctx = get_context("spawn")
    processes = []
    for proc_rank, gpu_id in enumerate(args._gpu_ids):
        p = mp_ctx.Process(
            target=infer_worker,
            args=(proc_rank, gpu_id, args, all_indices, save_path),
        )
        p.start()
        processes.append(p)

    failed = False
    for p in processes:
        p.join()
        if p.exitcode != 0:
            failed = True
            print(f"Process pid={p.pid} exit with code {p.exitcode}")

    if failed:
        raise RuntimeError("Some worker processes failed.")
    print("All workers finished successfully.")


if __name__ == "__main__":
    main()