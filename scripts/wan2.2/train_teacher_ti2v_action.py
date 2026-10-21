# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Teacher (bidirectional) action training for Wan2.2 TI2V.

The DMD teacher is the vanilla bidirectional Wan2.2-TI2V-5B base optionally merged with a
teacher LoRA. Its adaLN action module must be calibrated against THAT base + full
bidirectional attention -- a causal-trained action module (train_causal_ti2v_action.py)
does not transfer because its per-block adaLN scale/shift were tuned to the causal base and
chunk attention.

This script freezes the teacher (vanilla base + merged teacher LoRA) and trains ONLY the
action modules (action_preprocess + per-block action_adaLN_modulation) of
WanActionAdaLNModel, running full bidirectional TI2V diffusion. The resulting checkpoint is
what the DMD trainer consumes as its action-aware teacher (--teacher_action_pretrain_path).

Load order (teacher analog of the causal action trainer):
    vanilla bidirectional base -> teacher LoRA (frozen, merged) -> action (adaLN) init
Checkpoints are exported with the teacher LoRA subtracted, so a saved checkpoint holds only
(vanilla base + trained action) and the teacher LoRA can be re-merged at inference time.
"""
#!/usr/bin/env python
# coding=utf-8
try:
    import comet_ml
except ImportError:
    print("Comet is not installed in your env! Exp res will not be record.")

import gc
import logging
import math
import os
import random
import shutil
import sys
from collections import OrderedDict
from types import SimpleNamespace

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
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.training_utils import compute_loss_weighting_for_sd3
from diffusers.utils import check_min_version, is_wandb_available
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
import diffusers
from transformers import AutoTokenizer
from transformers.utils import ContextManagers
from datetime import datetime
import argparse

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None
from microworld.data.bucket_sampler import RandomSampler
from microworld.data.dataset_game_video import (VideoGameDataset,
                                                GameVideoSampler)
from microworld.models.wan_vae_3_8 import AutoencoderKLWan3_8
from microworld.models import AutoencoderKLWan, WanT5EncoderModel, WanActionAdaLNModel
from microworld.pipeline.pipeline_wan_action_ti2w import Wan2_2ActionTI2WPipeline
from microworld.utils.discrete_sampler import DiscreteSampling
from microworld.utils.lora_utils import merge_lora
from microworld.utils.utils import save_videos_grid, get_image_to_video_latent, parse_action_list
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


# Same fixed validation action script as the causal trainer, so teacher/causal action runs
# are directly comparable. Per frame: [w, s, a, d, shift, ctrl, _, mouse_y, mouse_x].
VALIDATION_ACTION_LIST = [
    [20,  "1 0 0 0 0 0 0 0.4 0"],    # W + pitch up
    [40,  "0 1 0 0 0 0 0 0 0.6"],    # S + yaw right
    [80,  "0 0 0 1 0 0 0 0 -0.6"],   # D + yaw left
    [110, "1 0 0 0 0 0 0 0 0.6"],    # W + yaw right
    [140, "0 0 1 0 0 0 0 0.4 0"],    # A + pitch up
    [170, "1 0 0 0 0 0 0 0 -0.6"],   # W + yaw left
    [196, "0 0 0 1 0 0 0 -0.4 0"],   # D + pitch down
    [225, "1 0 0 0 0 0 0 0 0.6"],    # W + yaw right
    [255, "0 0 1 0 0 0 0 -0.4 0"],   # A + pitch down
    [285, "1 0 0 0 0 0 0 0 -0.6"],   # W + yaw left
    [315, "0 1 0 0 0 0 0 0.4 0"],    # S + pitch up
    [345, "0 0 0 1 0 0 0 0 0.6"],    # D + yaw right
    [375, "1 0 0 0 0 0 0 0 -0.6"],   # W + yaw left
    [396, "1 0 0 0 0 0 0 -0.4 0"],   # W + pitch down
    "",
]


def _pad_or_truncate(t, length):
    if t.shape[0] >= length:
        return t[:length]
    pad = t[-1:].expand(length - t.shape[0], -1)
    return torch.cat([t, pad], dim=0)


def build_validation_actions(num_frames):
    keyboard_actions, mouse_actions = parse_action_list(VALIDATION_ACTION_LIST)
    keyboard_actions = _pad_or_truncate(keyboard_actions, num_frames)[None].float()
    mouse_actions = _pad_or_truncate(mouse_actions, num_frames)[None].float()
    return keyboard_actions, mouse_actions


@torch.no_grad()
def log_validation(vae, text_encoder, tokenizer, transformer3d, config, args, accelerator, weight_dtype, global_step):
    try:
        logger.info("Running validation... ")

        scheduler = FlowUniPCMultistepScheduler(
            **filter_kwargs(FlowUniPCMultistepScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
        )
        boundary = config['transformer_additional_kwargs'].get('boundary', 0.875)

        transformer3d = accelerator.unwrap_model(transformer3d)
        vae = accelerator.unwrap_model(vae)
        text_encoder = accelerator.unwrap_model(text_encoder)

        was_training = transformer3d.training
        transformer3d.eval()

        pipeline = Wan2_2ActionTI2WPipeline(
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
        input_video, input_video_mask, _ = get_image_to_video_latent(
            args.validation_image_start,
            None,
            video_length=validation_num_frames,
            sample_size=args.video_sample_size,
        )
        input_video = input_video.to(device=accelerator.device, dtype=weight_dtype)
        input_video_mask = torch.zeros_like(input_video[:, :1])
        input_video_mask[:, :, 1:] = 255

        text = args.validation_prompt
        video_length = int((validation_num_frames - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if validation_num_frames != 1 else 1
        keyboard_actions, mouse_actions = build_validation_actions(video_length)
        keyboard_actions = keyboard_actions.to(device=accelerator.device, dtype=weight_dtype)
        mouse_actions = mouse_actions.to(device=accelerator.device, dtype=weight_dtype)

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
                boundary=boundary,
                mouse_actions=mouse_actions,
                keyboard_actions=keyboard_actions,
                with_ui=True,
            ).videos

        os.makedirs(os.path.join(args.output_dir, "sample"), exist_ok=True)
        input_video = input_video[:, :, :sample.shape[2]].to(device=sample.device, dtype=sample.dtype)
        sample = torch.cat([sample, input_video], dim=-1)
        sample_path = os.path.join(args.output_dir, f"sample/sample-{global_step}.mp4")
        save_videos_grid(sample, sample_path, fps=args.validation_fps)
        with open(os.path.join(args.output_dir, "sample", "prompt.log"), "a") as f:
            f.write(f"{global_step}, {text}\n")

        del pipeline
        if was_training:
            transformer3d.train()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception as e:
        if "transformer3d" in locals() and "was_training" in locals() and was_training:
            transformer3d.train()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print(f"Eval error with info {e}")
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Teacher (bidirectional) action training for Wan2.2 TI2V.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None, required=True, help="Path to pretrained model.")
    parser.add_argument("--config_path", type=str, default="config/wan2.2/wan_civitai_5b.yaml", help="Path to Wan2.2 config.")
    parser.add_argument("--train_data_dir", type=str, default=None, help="A folder containing the training data.")
    parser.add_argument("--train_data_meta", type=str, default=None, help="A csv/json containing the training data.")
    parser.add_argument("--output_dir", type=str, default="output_dir_teacher_action", help="Output dir for checkpoints.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size per device.")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total training steps; overrides num_train_epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--scale_lr", action="store_true", default=False)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--video_sample_size", type=int, nargs="+", default=[352, 640])
    parser.add_argument("--video_sample_n_frames", type=int, default=81)
    parser.add_argument("--video_sample_stride", type=int, default=1)
    parser.add_argument("--text_drop_ratio", type=float, default=0.1)
    parser.add_argument("--drop_camera_sentences", action="store_true")
    parser.add_argument("--enable_random_temporal_sample", action="store_true")
    parser.add_argument("--temporal_reverse_ratio", type=float, default=0.0)
    parser.add_argument("--train_sampling_steps", type=int, default=1000)
    parser.add_argument("--uniform_sampling", action="store_true")
    parser.add_argument("--weighting_scheme", type=str, default="none", choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"])
    parser.add_argument('--trainable_modules', nargs='+', default=["action"], help='List of trainable module name substrings.')
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a transformer checkpoint dir, or 'latest'.")
    parser.add_argument("--teacher_lora_path", type=str, default=None, help="Teacher LoRA merged and frozen into the bidirectional base before action training.")
    parser.add_argument("--teacher_lora_multiplier", type=float, default=1.0)
    parser.add_argument("--action_pretrain_path", type=str, default=None, help="WanActionAdaLNModel (or compatible) checkpoint dir; only its action (adaLN) modules are loaded to warm-start.")
    parser.add_argument("--validation_steps", type=int, default=200)
    parser.add_argument("--validation_image_start", type=str, default="asset/cliff.jpg")
    parser.add_argument("--validation_prompt", type=str, default="First-person view walking forward along a stone-paved clifftop path by the sea.")
    parser.add_argument("--validation_negative_prompt", type=str, default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")
    parser.add_argument("--validation_num_inference_steps", type=int, default=30)
    parser.add_argument("--validation_guidance_scale", type=float, default=5.0)
    parser.add_argument("--validation_shift", type=float, default=5.0)
    parser.add_argument("--validation_fps", type=int, default=16)
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--vae_path", type=str, default=None)
    parser.add_argument("--vae_mini_batch", type=int, default=1)
    parser.add_argument("--i2v_ratio", type=float, default=1.0, help="Probability of using first frame as clean TI2V history (teacher eval is i2v, so default 1.0).")

    args = parser.parse_args()
    if len(args.video_sample_size) == 1:
        args.video_sample_size = [args.video_sample_size[0], args.video_sample_size[0]]
    return args


def _extract_arch_signature(model_or_config):
    config = model_or_config.config if hasattr(model_or_config, "config") else model_or_config
    keys = ("model_type", "in_dim", "out_dim", "dim", "num_heads", "num_layers")
    return {k: config.get(k, None) for k in keys}


def _assert_same_arch(expected, actual, source_name, ignore_keys=()):
    diffs = []
    for k, v in expected.items():
        if k in ignore_keys:
            continue
        if actual.get(k, None) != v:
            diffs.append(f"{k}: expected {v}, got {actual.get(k, None)}")
    if diffs:
        raise ValueError(
            f"Incompatible architecture from {source_name}. "
            + " ; ".join(diffs)
            + ". Please use a checkpoint with the same transformer shape."
        )


def _sync_model_config_from_weights(transformer3d):
    patch_w = transformer3d.patch_embedding.weight
    # The head projects to prod(patch_size) * out_dim, so divide that back out to recover
    # the out_dim the model has to be rebuilt with.
    patch_volume = int(math.prod(transformer3d.patch_size))
    head_out_features = getattr(transformer3d.head.head, "out_features", None)
    out_dim = int(head_out_features) // patch_volume if head_out_features else int(transformer3d.out_dim)
    transformer3d.register_to_config(
        model_type=getattr(transformer3d, "model_type", "t2v"),
        in_dim=int(patch_w.shape[1]),
        out_dim=out_dim,
        dim=int(patch_w.shape[0]),
        ffn_dim=int(transformer3d.blocks[0].ffn[0].out_features),
        num_heads=int(transformer3d.num_heads),
        num_layers=int(transformer3d.num_layers),
    )


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

    # Bidirectional teacher base with action modules. low_cpu_mem_usage=False so the action
    # modules (absent from the base checkpoint) get real storage for the later in-place loads.
    transformer3d = WanActionAdaLNModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['transformer_additional_kwargs'].get('transformer_low_noise_model_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=False,
        torch_dtype=weight_dtype,
    ).to(weight_dtype)
    expected_arch = _extract_arch_signature(transformer3d)
    accelerator.print(f"Loaded base transformer arch: {expected_arch}")

    # 1) Teacher LoRA merged (frozen) into the bidirectional base.
    if args.teacher_lora_path is not None:
        if not os.path.exists(args.teacher_lora_path):
            raise FileNotFoundError(f"Teacher LoRA checkpoint not found: {args.teacher_lora_path}")
        print(f"Merging frozen teacher LoRA from {args.teacher_lora_path}")
        merge_lora(
            SimpleNamespace(transformer=transformer3d, text_encoder=text_encoder),
            args.teacher_lora_path,
            args.teacher_lora_multiplier,
            device="cpu",
            dtype=torch.float32,
            transformer_only=True,
        )

    # 2) Action init: overlay ONLY the action (adaLN) modules, keeping the teacher base intact.
    if args.action_pretrain_path is not None:
        print(f"Loading action (adaLN) modules from {args.action_pretrain_path}")
        action_pretrain = WanActionAdaLNModel.from_pretrained(
            args.action_pretrain_path,
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
            low_cpu_mem_usage=True,
        ).to(weight_dtype)
        # Only the action (adaLN) modules are warm-started here (strict=False below), and those
        # depend on `dim`/`num_heads`/`num_layers`, not on in_dim/out_dim. Some action checkpoints
        # record out_dim as the flattened head width (out_channels*patch, e.g. 192) rather than the
        # channel count (48); that is a cosmetic config difference, so ignore in_dim/out_dim here.
        _assert_same_arch(
            expected_arch,
            _extract_arch_signature(action_pretrain),
            args.action_pretrain_path,
            ignore_keys=("in_dim", "out_dim"),
        )
        full_sd = action_pretrain.state_dict()
        action_sd = {
            k: v for k, v in full_sd.items()
            if k.startswith("action_preprocess") or "action_adaLN_modulation" in k
        }
        m, u = transformer3d.load_state_dict(action_sd, strict=False)
        print(f"loaded {len(action_sd)} action params; unexpected keys: {len(u)}")
        del action_pretrain, full_sd

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    transformer3d.requires_grad_(False)

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

    if accelerator.is_main_process:
        accelerator.print(f"Trainable modules '{args.trainable_modules}'.")
    for name, param in transformer3d.named_parameters():
        for trainable_module_name in args.trainable_modules:
            if trainable_module_name in name:
                param.requires_grad = True
                break

    trainable_params = list(filter(lambda p: p.requires_grad, transformer3d.parameters()))

    optimizer = optimizer_cls(
        trainable_params,
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
        use_action=True,
        enable_random_temporal_sample=args.enable_random_temporal_sample,
        temporal_reverse_ratio=args.temporal_reverse_ratio,
    )

    def collate_fn(examples):
        new_examples = {}
        new_examples["pixel_values"] = []
        new_examples["text"] = []
        new_examples["idx"] = []
        new_examples["mouse_actions"] = []
        new_examples["keyboard_actions"] = []

        for example in examples:
            new_examples["pixel_values"].append(example["pixel_values"])
            new_examples["text"].append(example["text"])
            new_examples["idx"].append(example["idx"])
            new_examples["mouse_actions"].append(example["mouse_actions"])
            new_examples["keyboard_actions"].append(example["keyboard_actions"])

        new_examples["pixel_values"] = torch.stack(new_examples["pixel_values"])  # [B, F, C, H, W]
        new_examples["idx"] = torch.tensor(new_examples["idx"], dtype=torch.long)
        new_examples["mouse_actions"] = torch.stack(new_examples["mouse_actions"])
        new_examples["keyboard_actions"] = torch.stack(new_examples["keyboard_actions"])
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

    transformer3d, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer3d, optimizer, train_dataloader, lr_scheduler
    )

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("wan2.2-teacher-action")

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_transformer_without_teacher_lora(save_dir):
        unwrapped_transformer = unwrap_model(transformer3d)
        _sync_model_config_from_weights(unwrapped_transformer)
        if args.teacher_lora_path is not None:
            # Subtract the merged teacher LoRA so the exported checkpoint holds only
            # (vanilla base + trained action); restore afterwards.
            mock_pipeline = SimpleNamespace(transformer=unwrapped_transformer)
            merge_lora(
                mock_pipeline,
                args.teacher_lora_path,
                -args.teacher_lora_multiplier,
                device="cpu",
                dtype=torch.float32,
                transformer_only=True,
            )
            unwrapped_transformer.save_pretrained(save_dir)
            merge_lora(
                mock_pipeline,
                args.teacher_lora_path,
                args.teacher_lora_multiplier,
                device="cpu",
                dtype=torch.float32,
                transformer_only=True,
            )
        else:
            unwrapped_transformer.save_pretrained(save_dir)

    resume_checkpoint_dir = None
    if args.resume_from_checkpoint:
        ckpt_dir = args.resume_from_checkpoint
        if ckpt_dir.split("/")[-1] == "latest":
            ckpt_dir = os.readlink(ckpt_dir)
        resume_checkpoint_dir = ckpt_dir
        global_step = int(ckpt_dir.split("-")[-1])
        first_epoch = global_step // num_update_steps_per_epoch
        accelerator.print(f"Resuming from checkpoint {ckpt_dir}")
        load_model = WanActionAdaLNModel.from_pretrained(
            ckpt_dir,
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        )
        _assert_same_arch(expected_arch, _extract_arch_signature(load_model), ckpt_dir)
        unwrapped_transformer3d = unwrap_model(transformer3d)
        unwrapped_transformer3d.register_to_config(**load_model.config)
        m, u = unwrapped_transformer3d.load_state_dict(load_model.state_dict(), strict=False)
        accelerator.print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        del load_model
        if args.teacher_lora_path is not None:
            # Checkpoints are exported with teacher LoRA subtracted. Re-apply it on resume.
            accelerator.print(f"Re-merging frozen teacher LoRA from {args.teacher_lora_path}")
            merge_lora(
                SimpleNamespace(transformer=unwrapped_transformer3d, text_encoder=text_encoder),
                args.teacher_lora_path,
                args.teacher_lora_multiplier,
                device="cpu",
                dtype=torch.float32,
                transformer_only=True,
            )
        optimizer_path = os.path.join(ckpt_dir, "optimizer.pt")
        scheduler_path = os.path.join(ckpt_dir, "scheduler.pt")
        if os.path.exists(optimizer_path):
            accelerator.print(f"Loading optimizer state from {optimizer_path}")
            optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))
        else:
            accelerator.print(f"[Warning] Optimizer state not found at {optimizer_path}.")
        if os.path.exists(scheduler_path):
            accelerator.print(f"Loading LR scheduler state from {scheduler_path}")
            lr_scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu"))
        else:
            accelerator.print(f"[Warning] LR scheduler state not found at {scheduler_path}.")
    else:
        global_step = 0
        first_epoch = 0

    if accelerator.is_main_process and args.validation_steps > 0:
        log_validation(vae, text_encoder, tokenizer, transformer3d, config, args, accelerator, weight_dtype, global_step)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

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
    resume_batches_to_skip = 0
    if resume_checkpoint_dir is not None:
        resume_step_in_epoch = global_step % num_update_steps_per_epoch
        resume_batches_to_skip = resume_step_in_epoch * args.gradient_accumulation_steps
        accelerator.print(
            f"Resume data position: first_epoch={first_epoch}, skip_batches_in_first_epoch={resume_batches_to_skip}"
        )

    sample_n_frames_bucket_interval = vae.config.temporal_compression_ratio

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        batch_sampler.sampler.generator = torch.Generator().manual_seed((args.seed or 0) + epoch)
        for step, batch in enumerate(train_dataloader):
            if epoch == first_epoch and step < resume_batches_to_skip:
                continue
            if epoch == first_epoch and step == 0 and accelerator.is_main_process:
                pixel_values, texts = batch['pixel_values'].cpu(), batch['text']
                pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                os.makedirs(os.path.join(args.output_dir, "sanity_check"), exist_ok=True)
                for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                    pixel_value = pixel_value[None, ...]
                    gif_name = '-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_step}-{idx}'
                    save_videos_grid(pixel_value, f"{args.output_dir}/sanity_check/{gif_name[:10]}.gif", rescale=True)

            with accelerator.accumulate(transformer3d):
                pixel_values = batch["pixel_values"].to(device=accelerator.device, dtype=weight_dtype)  # [B, F, C, H, W]
                mouse_actions = batch["mouse_actions"].to(device=accelerator.device, dtype=weight_dtype)
                keyboard_actions = batch["keyboard_actions"].to(device=accelerator.device, dtype=weight_dtype)

                # Build the i2v inpaint mask: first frame is the clean reference (mask=0).
                bsz_px, f_px, c_px, h_px, w_px = pixel_values.size()
                mask = torch.ones((bsz_px, f_px, 1, h_px, w_px), device=accelerator.device, dtype=weight_dtype)
                mask[:, 0] = 0
                mask_pixel_values = pixel_values * (1 - mask)

                with torch.no_grad():
                    def _batch_encode_vae(px):
                        px = rearrange(px, "b f c h w -> b c f h w")
                        bs = args.vae_mini_batch
                        out = []
                        for i in range(0, px.shape[0], bs):
                            px_bs = px[i:i + bs]
                            px_bs = vae.encode(px_bs)[0].sample()
                            out.append(px_bs)
                        return torch.cat(out, dim=0)

                    latents = _batch_encode_vae(pixel_values)

                    # Project the pixel-space mask onto the latent temporal/spatial grid, matching
                    # the original bidirectional action trainer.
                    mask_lat = rearrange(mask, "b f c h w -> b c f h w")
                    mask_lat = torch.concat(
                        [torch.repeat_interleave(mask_lat[:, :, 0:1], repeats=sample_n_frames_bucket_interval, dim=2), mask_lat[:, :, 1:]],
                        dim=2,
                    )
                    mask_lat = rearrange(mask_lat, "b 1 (f d) h w -> b d f h w", d=sample_n_frames_bucket_interval)
                    mask_lat = F.interpolate(mask_lat[:, :1], size=latents.size()[-3:], mode='trilinear', align_corners=True).to(accelerator.device, weight_dtype)
                    inpaint_latents = _batch_encode_vae(mask_pixel_values)

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

                patch_h = accelerator.unwrap_model(transformer3d).config.patch_size[1]
                patch_w = accelerator.unwrap_model(transformer3d).config.patch_size[2]
                seq_len = math.ceil((height // patch_h) * (width // patch_w) * num_frames)

                # i2v vs t2v for this batch. For i2v, replace the clean region with the inpaint
                # latents and feed per-token timesteps (0 on the clean first-frame tokens).
                use_i2v = (random.random() < args.i2v_ratio) and num_frames > 1
                if use_i2v:
                    noisy_latents = (1 - mask_lat) * inpaint_latents + mask_lat * noisy_latents
                    temp_ts = (mask_lat[:, 0, :, ::patch_h, ::patch_w] * timesteps[:, None, None, None]).flatten(1)
                    if temp_ts.size(1) < seq_len:
                        pad = temp_ts.new_ones(bsz, seq_len - temp_ts.size(1)) * timesteps[:, None]
                        t_tokens = torch.cat([temp_ts, pad], dim=1)
                    else:
                        t_tokens = temp_ts[:, :seq_len]
                    loss_mask = mask_lat
                else:
                    t_tokens = mask_lat.new_ones(bsz, seq_len) * timesteps[:, None]
                    loss_mask = torch.ones_like(mask_lat)

                with accelerator.autocast():
                    noise_pred = transformer3d(
                        x=noisy_latents,
                        context=prompt_embeds,
                        t=t_tokens,
                        seq_len=seq_len,
                        mouse_actions=mouse_actions,
                        keyboard_actions=keyboard_actions,
                    )

                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                mse_loss = F.mse_loss(noise_pred.float(), target.float(), reduction='none')
                mse_loss = mse_loss * loss_mask.float()
                if weighting is not None:
                    mse_loss = mse_loss * weighting.float()
                loss = mse_loss.sum() / loss_mask.float().expand_as(mse_loss).sum().clamp(min=1.0)

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()

                # NaN-guard: skip backward/step on a non-finite loss so a single numerical spike
                # can't permanently poison the weights (avg_loss is gathered => in sync on all ranks).
                if torch.isfinite(avg_loss):
                    train_loss += avg_loss.item() / args.gradient_accumulation_steps
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                else:
                    optimizer.zero_grad()
                    lr_scheduler.step()
                    if accelerator.is_main_process:
                        logger.warning(
                            f"[nan-guard] non-finite loss ({avg_loss.item()}) at step {global_step}; skipping optimizer.step. text={batch.get('text')}"
                        )

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
                                logger.info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")
                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        accelerator_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        save_transformer_without_teacher_lora(accelerator_save_path)
                        torch.save(optimizer.state_dict(), os.path.join(accelerator_save_path, "optimizer.pt"))
                        torch.save(lr_scheduler.state_dict(), os.path.join(accelerator_save_path, "scheduler.pt"))
                        logger.info(f"Saved state to {accelerator_save_path}")

                if accelerator.is_main_process:
                    if args.validation_steps > 0 and global_step % args.validation_steps == 0:
                        log_validation(vae, text_encoder, tokenizer, transformer3d, config, args, accelerator, weight_dtype, global_step)

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        save_transformer_without_teacher_lora(save_path)
        torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))
        torch.save(lr_scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))
        logger.info(f"Saved state to {save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
