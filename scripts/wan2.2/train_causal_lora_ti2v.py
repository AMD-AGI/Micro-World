# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Causal LoRA training for Wan2.2 TI2V.

This script follows the style of train_game_lora_ti2v.py, but trains only LoRA
parameters on CausalWanTransformer3DModel. TI2V is represented as a clean first
latent frame plus block-causal teacher forcing, without the original TI2V mask
mechanism.
"""
#!/usr/bin/env python
# coding=utf-8
try:
    import comet_ml
except ImportError:
    print("Comet is not installed in your env! Exp res will not be record.")

import argparse
from collections import OrderedDict
import gc
import logging
import math
import os
import random
import shutil
import sys

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_loss_weighting_for_sd3
from diffusers.utils import check_min_version, is_wandb_available
from einops import rearrange
from omegaconf import OmegaConf
from packaging import version
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
import diffusers
from transformers import AutoTokenizer
from transformers.utils import ContextManagers
from datetime import datetime

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None
from microworld.data.bucket_sampler import RandomSampler
from microworld.data.dataset_game_video import (VideoGameDataset,
                                                GameVideoSampler)
from microworld.models.wan_vae_3_8 import AutoencoderKLWan3_8
from microworld.models import AutoencoderKLWan, WanT5EncoderModel
from microworld.models.wan_causal_transformer3d import CausalWanTransformer3DModel
from microworld.pipeline.pipeline_wan_causal import WanCausalPipeline
from microworld.utils.discrete_sampler import DiscreteSampling
from microworld.utils import lora_utils
from microworld.utils.lora_utils import create_network
from microworld.utils.utils import save_videos_grid, get_image_to_video_latent
from microworld.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

if is_wandb_available():
    import wandb


def filter_kwargs(cls, kwargs):
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs


check_min_version("0.18.0.dev0")

logger = get_logger(__name__, log_level="INFO")


@torch.no_grad()
def log_validation(vae, text_encoder, tokenizer, transformer3d, network, config, args, accelerator, weight_dtype, global_step):
    try:
        logger.info("Running validation... ")

        scheduler = FlowUniPCMultistepScheduler(
            **filter_kwargs(FlowUniPCMultistepScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
        )

        transformer3d = accelerator.unwrap_model(transformer3d)
        network = accelerator.unwrap_model(network)
        vae = accelerator.unwrap_model(vae)
        text_encoder = accelerator.unwrap_model(text_encoder)

        was_training = transformer3d.training
        network_was_training = network.training
        transformer3d.eval()
        network.eval()

        pipeline = WanCausalPipeline(
            transformer=transformer3d,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            scheduler=scheduler,
        )
        pipeline = pipeline.to(accelerator.device)

        if args.seed is None:
            generator = None
        else:
            generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

        validation_num_frames = args.video_sample_n_frames
        video_length = int((validation_num_frames - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if validation_num_frames != 1 else 1
        latent_frames = (video_length - 1) // vae.config.temporal_compression_ratio + 1

        # Support multiple validation cases: pair each image with its prompt.
        validation_images = args.validation_image_start
        if isinstance(validation_images, str):
            validation_images = [validation_images]
        validation_prompts = args.validation_prompt
        if isinstance(validation_prompts, str):
            validation_prompts = [validation_prompts]
        if len(validation_prompts) == 1 and len(validation_images) > 1:
            validation_prompts = validation_prompts * len(validation_images)

        os.makedirs(os.path.join(args.output_dir, "sample"), exist_ok=True)

        for case_idx, (image_start, text) in enumerate(zip(validation_images, validation_prompts)):
            input_video, input_video_mask, _ = get_image_to_video_latent(
                image_start,
                None,
                video_length=validation_num_frames,
                sample_size=args.video_sample_size,
            )
            input_video = input_video.to(device=accelerator.device, dtype=weight_dtype)
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, 1:] = 255

            if args.seed is not None:
                generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

            with accelerator.autocast():
                sample = pipeline(
                    text,
                    num_frames=video_length,
                    negative_prompt=args.validation_negative_prompt,
                    height=args.video_sample_size[0],
                    width=args.video_sample_size[1],
                    generator=generator,
                    guidance_scale=args.validation_guidance_scale,
                    num_inference_steps=args.validation_num_inference_steps,
                    video=input_video[:, :, :video_length],
                    mask_video=input_video_mask[:, :, :video_length],
                    shift=args.validation_shift,
                    chunk_size=args.chunk_size,
                ).videos

            sample_path = os.path.join(args.output_dir, f"sample/sample-{global_step}-{case_idx}.mp4")
            all_sample = torch.cat([sample, input_video[:, :, :sample.shape[2]].to(sample.device, sample.dtype)], dim=-1)
            save_videos_grid(all_sample, sample_path, fps=args.validation_fps)
            with open(os.path.join(args.output_dir, "sample", "prompt.log"), "a") as f:
                f.write(f"{global_step}, case={case_idx}, latent_frames={latent_frames}, {os.path.basename(image_start)}, {text}\n")

        del pipeline
        if was_training:
            transformer3d.train()
        if network_was_training:
            network.train()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception as e:
        if "transformer3d" in locals() and "was_training" in locals() and was_training:
            transformer3d.train()
        if "network" in locals() and "network_was_training" in locals() and network_was_training:
            network.train()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print(f"Eval error with info {e}")
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Causal LoRA training for Wan2.2 TI2V.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/wan2.2/wan_civitai_5b.yaml",
        help="Path to Wan2.2 config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data.",
    )
    parser.add_argument(
        "--train_data_meta",
        type=str,
        default=None,
        help="A csv/json containing the training data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="wan2.2-causal-lora",
        help="The output directory where checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size per device."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale lr by number of GPUs, gradient accumulation, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help="The scheduler type to use.",
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=0, help="Number of steps for lr warmup."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for Adam.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for Adam.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses for data loading.",
    )
    parser.add_argument(
        "--video_sample_size",
        type=int,
        nargs="+",
        default=[352, 640],
        help="Video sample size.",
    )
    parser.add_argument(
        "--video_sample_n_frames",
        type=int,
        default=81,
        help="Number of pixel frames sampled from each video.",
    )
    parser.add_argument(
        "--video_sample_stride",
        type=int,
        default=1,
        help="Video frame stride.",
    )
    parser.add_argument(
        "--text_drop_ratio",
        type=float,
        default=0.1,
        help="Drop text ratio in dataset.",
    )
    parser.add_argument(
        "--drop_camera_sentences",
        action="store_true",
        help="Drop caption sentences containing camera/motion descriptions.",
    )
    parser.add_argument(
        "--train_sampling_steps",
        type=int,
        default=1000,
        help="Number of training timestep buckets.",
    )
    parser.add_argument(
        "--uniform_sampling",
        action="store_true",
        help="Whether to use distributed uniform timestep sampling.",
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help="Loss weighting scheme.",
    )
    parser.add_argument("--rank", type=int, default=128, help="The dimension of the LoRA update matrices.")
    parser.add_argument("--network_alpha", type=int, default=64, help="The alpha of the LoRA update matrices.")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save a checkpoint every X updates.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="Max number of checkpoints to store.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a LoRA checkpoint directory or lora_diffusion_pytorch_model.safetensors.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=200,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--validation_image_start",
        type=str,
        nargs="+",
        default=["asset/street_night.jpg", "asset/first_frame.png"],
        help="Reference image path(s) for TI2V validation. Accepts multiple, paired with --validation_prompt.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        nargs="+",
        default=["First-person perspective walking in a lively city street at night."],
        help="Prompt(s) used in validation inference. Provide one per image (or a single prompt for all).",
    )
    parser.add_argument(
        "--validation_negative_prompt",
        type=str,
        default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        help="Negative prompt used in validation inference.",
    )
    parser.add_argument(
        "--validation_num_inference_steps",
        type=int,
        default=30,
        help="Inference steps used in validation generation.",
    )
    parser.add_argument(
        "--validation_guidance_scale",
        type=float,
        default=6.0,
        help="Guidance scale used in validation generation.",
    )
    parser.add_argument(
        "--validation_shift",
        type=float,
        default=5.0,
        help="Shift parameter used in validation generation.",
    )
    parser.add_argument(
        "--validation_fps",
        type=int,
        default=16,
        help="FPS used when saving validation videos.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision.",
    )
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help="Optional VAE checkpoint.",
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        default=None,
        help="Optional transformer checkpoint.",
    )
    parser.add_argument(
        "--vae_mini_batch",
        type=int,
        default=1,
        help="Mini batch size for VAE encoding.",
    )
    parser.add_argument(
        "--i2v_ratio",
        type=float,
        default=0.5,
        help="Probability of using first latent frame as clean TI2V history.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=3,
        help="Latent frames per causal chunk after the independent first frame.",
    )
    parser.add_argument(
        "--independent_first_frame",
        action="store_true",
        default=True,
        help="Use [1, N, N, ...] block layout for causal training.",
    )
    parser.add_argument(
        "--enable_text_encoder_in_dataloader",
        action="store_true",
        help="Compatibility flag; this script expects raw text by default.",
    )

    args = parser.parse_args()
    if len(args.video_sample_size) == 1:
        args.video_sample_size = [args.video_sample_size[0], args.video_sample_size[0]]
    return args


def configure_causal_teacher_forcing(transformer3d, chunk_size, independent_first_frame):
    transformer3d.chunk_size = chunk_size
    transformer3d.independent_first_frame = independent_first_frame
    for block in transformer3d.blocks:
        block.self_attn.chunk_size = chunk_size
        block.self_attn.independent_first_frame = independent_first_frame


def resolve_lora_checkpoint_path(resume_from_checkpoint):
    checkpoint_path = resume_from_checkpoint
    if checkpoint_path.split("/")[-1] == "latest":
        checkpoint_path = os.readlink(checkpoint_path)
    if os.path.isdir(checkpoint_path):
        checkpoint_path = os.path.join(checkpoint_path, "lora_diffusion_pytorch_model.safetensors")
    return checkpoint_path


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


def main():
    args = parse_args()

    now = datetime.now()
    time_str = now.strftime("%Y%m%d-%H%M")
    args.output_dir = os.path.join(args.output_dir, time_str)
    logging_dir = os.path.join(args.output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=accelerator_project_config,
    )
    deepspeed_plugin = accelerator.state.deepspeed_plugin
    if deepspeed_plugin is not None:
        zero_stage = int(deepspeed_plugin.zero_stage)
        print(f"Using DeepSpeed Zero stage: {zero_stage}")
    else:
        zero_stage = 0
        print("DeepSpeed is not enabled.")
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    config = OmegaConf.load(args.config_path)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )

    def deepspeed_zero_init_disabled_context_manager():
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = WanT5EncoderModel.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
            additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
        text_encoder = text_encoder.eval()
        vae = AutoencoderKLWan3_8.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, config['vae_kwargs'].get('vae_subpath', 'vae')),
            additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
        )
        vae = vae.eval()

    transformer3d = CausalWanTransformer3DModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['transformer_additional_kwargs'].get('transformer_low_noise_model_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    ).to(weight_dtype)
    configure_causal_teacher_forcing(transformer3d, args.chunk_size, args.independent_first_frame)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    transformer3d.requires_grad_(False)

    if "CausalWanTransformer3DModel" not in lora_utils.LoRANetwork.TRANSFORMER_TARGET_REPLACE_MODULE:
        lora_utils.LoRANetwork.TRANSFORMER_TARGET_REPLACE_MODULE.append("CausalWanTransformer3DModel")

    network = create_network(
        1.0,
        args.rank,
        args.network_alpha,
        text_encoder,
        transformer3d,
        neuron_dropout=None,
    )
    network.apply_to(text_encoder, transformer3d, False, True)

    if args.transformer_path is not None:
        print(f"From checkpoint: {args.transformer_path}")
        if args.transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(args.transformer_path)
        else:
            state_dict = torch.load(args.transformer_path, map_location="cpu")
        state_dict = _unwrap_transformer_state_dict(state_dict)
        state_dict = _normalize_transformer_state_dict_keys(state_dict)
        m, u = transformer3d.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    if args.vae_path is not None:
        print(f"From checkpoint: {args.vae_path}")
        if args.vae_path.endswith("safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(args.vae_path)
        else:
            state_dict = torch.load(args.vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        m, u = vae.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    if args.gradient_checkpointing:
        transformer3d.enable_gradient_checkpointing()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    optimizer_cls = torch.optim.AdamW
    logging.info("Add network parameters")
    trainable_params = list(filter(lambda p: p.requires_grad, network.parameters()))
    trainable_params_optim = network.prepare_optimizer_params(None, args.learning_rate, args.learning_rate)
    optimizer = optimizer_cls(
        trainable_params_optim,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = VideoGameDataset(
        ann_path=args.train_data_meta,
        data_root=args.train_data_dir,
        video_sample_size=args.video_sample_size,
        video_sample_stride=args.video_sample_stride,
        video_sample_n_frames=args.video_sample_n_frames,
        text_drop_ratio=args.text_drop_ratio,
        drop_camera_sentences=args.drop_camera_sentences,
        use_action=False,
    )

    def collate_fn(examples):
        new_examples = {}
        new_examples["pixel_values"] = []
        new_examples["text"] = []
        new_examples["idx"] = []

        for example in examples:
            new_examples["pixel_values"].append(example["pixel_values"])
            new_examples["text"].append(example["text"])
            new_examples["idx"].append(example["idx"])

        new_examples["pixel_values"] = torch.stack(new_examples["pixel_values"])
        new_examples["idx"] = torch.tensor(new_examples["idx"], dtype=torch.long)
        return new_examples

    train_sampler = RandomSampler(train_dataset)
    batch_sampler = GameVideoSampler(train_sampler, train_dataset, args.train_batch_size, drop_last=True)
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        network, optimizer, train_dataloader, lr_scheduler
    )

    vae.to(accelerator.device, dtype=weight_dtype)
    transformer3d.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("wan2.2-causal-lora")
        if args.validation_steps > 0:
            log_validation(
                vae,
                text_encoder,
                tokenizer,
                transformer3d,
                network,
                config,
                args,
                accelerator,
                weight_dtype,
                0,
            )

    if args.resume_from_checkpoint is not None:
        lora_checkpoint_path = resolve_lora_checkpoint_path(args.resume_from_checkpoint)
        if not os.path.exists(lora_checkpoint_path):
            raise FileNotFoundError(f"LoRA checkpoint not found: {lora_checkpoint_path}")
        from safetensors.torch import load_file
        accelerator.print(f"Loading LoRA checkpoint from {lora_checkpoint_path}")
        state_dict = load_file(lora_checkpoint_path, device=str(accelerator.device))
        m, u = accelerator.unwrap_model(network).load_state_dict(state_dict, strict=False)
        accelerator.print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    torch_rng = torch.Generator(device=accelerator.device)
    if args.seed is not None:
        torch_rng.manual_seed(args.seed)
    idx_sampling = DiscreteSampling(args.train_sampling_steps, uniform_sampling=args.uniform_sampling)

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        batch_sampler.sampler.generator = torch.Generator().manual_seed((args.seed or 0) + epoch)
        for step, batch in enumerate(train_dataloader):
            if epoch == first_epoch and step == 0 and accelerator.is_main_process:
                pixel_values, texts = batch['pixel_values'].cpu(), batch['text']
                pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                os.makedirs(os.path.join(args.output_dir, "sanity_check"), exist_ok=True)
                for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                    pixel_value = pixel_value[None, ...]
                    gif_name = '-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_step}-{idx}'
                    save_videos_grid(pixel_value, f"{args.output_dir}/sanity_check/{gif_name[:10]}.gif", rescale=True)

            with accelerator.accumulate(network):
                pixel_values = batch["pixel_values"].to(device=accelerator.device, dtype=weight_dtype)

                with torch.no_grad():
                    def _batch_encode_vae(pixel_values):
                        pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                        bs = args.vae_mini_batch
                        new_pixel_values = []
                        for i in range(0, pixel_values.shape[0], bs):
                            pixel_values_bs = pixel_values[i : i + bs]
                            pixel_values_bs = vae.encode(pixel_values_bs)[0]
                            pixel_values_bs = pixel_values_bs.sample()
                            new_pixel_values.append(pixel_values_bs)
                        return torch.cat(new_pixel_values, dim = 0)

                    latents = _batch_encode_vae(pixel_values)

                with torch.no_grad():
                    prompt_ids = tokenizer(
                        batch['text'],
                        padding="max_length",
                        max_length=config['text_encoder_kwargs'].get('model_max_length', 512),
                        truncation=True,
                        add_special_tokens=True,
                        return_tensors="pt",
                    )
                    text_input_ids = prompt_ids.input_ids
                    prompt_attention_mask = prompt_ids.attention_mask

                    seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
                    prompt_embeds = text_encoder(text_input_ids.to(latents.device), attention_mask=prompt_attention_mask.to(latents.device))[0]
                    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]

                bsz, channel, num_frames, height, width = latents.size()
                noise = torch.randn(latents.size(), device=latents.device, generator=torch_rng, dtype=weight_dtype)

                indices = idx_sampling(bsz, generator=torch_rng, device=latents.device)
                indices = indices.long().cpu()
                timesteps = noise_scheduler.timesteps[indices].to(device=latents.device)

                def get_sigmas(timesteps, dtype=torch.float32):
                    sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
                    schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
                    timesteps = timesteps.to(accelerator.device)

                    step_indices = torch.searchsorted(schedule_timesteps.flip(0), timesteps)
                    step_indices = len(schedule_timesteps) - 1 - step_indices
                    sigmas = sigmas[step_indices]
                    sigmas = sigmas.view(bsz, 1, 1, 1, 1)
                    return sigmas

                sigmas = get_sigmas(timesteps, dtype=latents.dtype)
                noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
                target = noise - latents

                use_i2v = random.random() < args.i2v_ratio
                loss_mask = torch.ones((bsz, 1, num_frames, 1, 1), device=latents.device, dtype=weight_dtype)
                timestep_frames = timesteps[:, None].repeat(1, num_frames)
                if use_i2v and num_frames > 1:
                    noisy_latents[:, :, :1] = latents[:, :, :1]
                    timestep_frames[:, 0] = 0
                    loss_mask[:, :, :1] = 0

                patch_h = accelerator.unwrap_model(transformer3d).config.patch_size[1]
                patch_w = accelerator.unwrap_model(transformer3d).config.patch_size[2]
                seq_len = math.ceil((height // patch_h) * (width // patch_w) * num_frames)

                with accelerator.autocast():
                    noise_pred = transformer3d(
                        x=noisy_latents,
                        context=prompt_embeds,
                        t=timestep_frames,
                        seq_len=seq_len,
                        clean_x=latents,
                    )

                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                mse_loss = F.mse_loss(noise_pred.float(), target.float(), reduction='none')
                mse_loss = mse_loss * loss_mask.float()
                if weighting is not None:
                    mse_loss = mse_loss * weighting.float()
                loss = mse_loss.sum() / loss_mask.float().expand_as(mse_loss).sum().clamp(min=1.0)

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]
                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        accelerator.unwrap_model(network).save_weights(
                            os.path.join(save_path, "lora_diffusion_pytorch_model.safetensors"),
                            dtype=torch.float32,
                            metadata={},
                        )
                        logger.info(f"Saved state to {save_path}")

                if accelerator.is_main_process:
                    if args.validation_steps > 0 and global_step % args.validation_steps == 0:
                        log_validation(
                            vae,
                            text_encoder,
                            tokenizer,
                            transformer3d,
                            network,
                            config,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                        )

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        accelerator.unwrap_model(network).save_weights(
            os.path.join(args.output_dir, "lora_diffusion_pytorch_model.safetensors"),
            dtype=torch.float32,
            metadata={},
        )
    accelerator.end_training()


if __name__ == "__main__":
    main()
