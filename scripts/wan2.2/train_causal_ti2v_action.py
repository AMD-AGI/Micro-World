# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Causal action training for Wan2.2 TI2V.

This script can merge a frozen causal TI2V LoRA trained by train_causal_lora_ti2v.py,
then trains only action_preprocess and action AdaLN modulation parameters on
CausalWanActionAdaLNModel.
"""
#!/usr/bin/env python
# coding=utf-8
try:
    import comet_ml
except ImportError:
    print("Comet is not installed in your env! Exp res will not be record.")

import argparse
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
from microworld.models.wan_causal_adaln_action import CausalWanActionAdaLNModel
from microworld.pipeline.pipeline_wan_causal import WanCausalPipeline
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


# Mirror the inference eval "default" preset (build_action_presets in
# examples/wan2.2/predict_causal_action_evalcase.py) so step-0 validation matches inference.
# Per frame: [w, s, a, d, shift, ctrl, _, mouse_y, mouse_x]; trailing string = jump (space) frames.
# Mouse magnitudes are in the corrected (degrees) scale. Tuned to the capsaicin 200f training
# distribution so validation motion matches what the model saw in training: per-frame yaw
# nonzero-median ~0.62 (p95 ~0.77), pitch nonzero-median ~0.43 (p95 ~0.50). We use yaw ~0.6
# and pitch ~0.4 (near each median) so the validation pans/tilts are in-distribution.
# Covers frames 1..396 (so up to a 397-frame validation clip keeps moving the whole way instead
# of freezing on the last action). Motion stays in-distribution (yaw ~0.6, pitch ~0.4) and
# alternates keyboard + camera direction so long-horizon rollout is actually exercised. For
# shorter clips (e.g. 197) the tail is simply truncated, so this stays backward-compatible.
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


def build_validation_actions(num_frames, keyboard_dim=7, mouse_scale=0.6):
    # mouse_scale is unused here on purpose: the inference eval case feeds raw
    # mouse deltas from VALIDATION_ACTION_LIST, so we mirror that exactly.
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

        transformer3d = accelerator.unwrap_model(transformer3d)
        vae = accelerator.unwrap_model(vae)
        text_encoder = accelerator.unwrap_model(text_encoder)

        was_training = transformer3d.training
        transformer3d.eval()
        # Validation path always uses i2v-style conditioning (first frame as reference),
        # so force the causal layout to i2v for stable chunk behavior.
        configure_causal_teacher_forcing(
            transformer3d,
            args.chunk_size,
            True,
            args.local_attn_size,
            args.sink_size,
        )

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
        latent_frames = (video_length - 1) // vae.config.temporal_compression_ratio + 1
        keyboard_actions, mouse_actions = build_validation_actions(
            video_length,
            mouse_scale=args.validation_mouse_scale,
        )
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
                chunk_size=args.chunk_size,
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
            f.write(f"{global_step}, latent_frames={latent_frames}, {text}\n")

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
    parser = argparse.ArgumentParser(description="Causal action LoRA training for Wan2.2 TI2V.")
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
        "--enable_random_temporal_sample",
        action="store_true",
        help="Randomly sample the temporal window within each clip instead of always "
             "starting at the CSV start frame (data augmentation for long clips).",
    )
    parser.add_argument(
        "--temporal_reverse_ratio",
        type=float,
        default=0.0,
        help="Probability of playing a sampled clip backwards (with actions reversed "
             "accordingly) as data augmentation. Only valid for static scenes where the "
             "sole motion is the first-person camera. 0 disables it.",
    )
    parser.add_argument(
        "--local_attn_size",
        type=int,
        default=-1,
        help="Local attention size.",
    )
    parser.add_argument(
        "--sink_size",
        type=int,
        default=0,
        help="Sink size.",
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
    parser.add_argument(
        '--trainable_modules', 
        nargs='+', 
        help='Enter a list of trainable modules'
    )
    parser.add_argument("--rank", type=int, default=128, help="Compatibility argument; action-only training does not create a new LoRA.")
    parser.add_argument("--network_alpha", type=int, default=64, help="Compatibility argument; action-only training does not create a new LoRA.")
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
        help="Path to a full transformer checkpoint directory or transformer_model.safetensors.",
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        default=None,
        help="Causal transformer init weights (CAUSAL_MODEL_PT) the base LoRA was trained on. Loaded into base before merging the LoRA.",
    )
    parser.add_argument(
        "--base_lora_path",
        type=str,
        default=None,
        help="Path to causal TI2V LoRA trained by train_causal_lora_ti2v.py. It is merged and frozen before action training.",
    )
    parser.add_argument(
        "--action_pretrain_path",
        type=str,
        default=None,
        help="Path to a pretrained WanActionAdaLNModel transformer directory; only its action (adaLN) modules are loaded for initialization.",
    )
    parser.add_argument(
        "--base_lora_multiplier",
        type=float,
        default=1.0,
        help="Multiplier for merging the frozen base causal TI2V LoRA.",
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
        default="asset/first_frame.png",
        help="Reference image path for TI2V validation.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default="First-person perspective walking in a lively city street at night.",
        help="Prompt used in validation inference.",
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
        "--validation_mouse_scale",
        type=float,
        default=0.6,
        help="Mouse yaw amplitude used in validation action script.",
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


def configure_causal_teacher_forcing(transformer3d, chunk_size, independent_first_frame, local_attn_size=-1, sink_size=0):
    transformer3d.chunk_size = chunk_size
    transformer3d.independent_first_frame = independent_first_frame
    for block in transformer3d.blocks:
        block.self_attn.chunk_size = chunk_size
        block.self_attn.independent_first_frame = independent_first_frame
        block.self_attn.local_attn_size = local_attn_size
        block.self_attn.sink_size = sink_size

def _extract_arch_signature(model_or_config):
    config = model_or_config.config if hasattr(model_or_config, "config") else model_or_config
    keys = ("model_type", "in_dim", "out_dim", "dim", "num_heads", "num_layers")
    return {k: config.get(k, None) for k in keys}


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


def set_batch_causal_layout(transformer3d, chunk_size, use_i2v, local_attn_size=-1, sink_size=0):
    """
    Dynamically switch causal layout by current task type:
      - i2v: independent_first_frame=True  -> [1, N, N, ...]
      - t2v: independent_first_frame=False -> [N, N, ...]
    """
    target = bool(use_i2v)
    if getattr(transformer3d, "independent_first_frame", None) == target:
        return
    configure_causal_teacher_forcing(
        transformer3d,
        chunk_size=chunk_size,
        independent_first_frame=target,
        local_attn_size=local_attn_size,
        sink_size=sink_size,
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

    transformer3d = CausalWanActionAdaLNModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['transformer_additional_kwargs'].get('transformer_low_noise_model_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    ).to(weight_dtype)
    expected_arch = _extract_arch_signature(transformer3d)
    accelerator.print(f"Loaded base transformer arch: {expected_arch}")
    configure_causal_teacher_forcing(transformer3d, args.chunk_size, args.independent_first_frame, args.local_attn_size, args.sink_size)

    # Load order mirrors inference (predict_causal_action_evalcase.py):
    #   base -> CAUSAL init (model.pt) -> base LoRA -> action (adaLN only).
    # 1) CAUSAL init weights (the base the LoRA was trained on).
    if args.transformer_path is not None:
        print(f"Loading causal transformer init from {args.transformer_path}")
        if args.transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(args.transformer_path)
        else:
            state_dict = torch.load(args.transformer_path, map_location="cpu")
        state_dict = _unwrap_transformer_state_dict(state_dict)
        state_dict = _normalize_transformer_state_dict_keys(state_dict)
        m, u = transformer3d.load_state_dict(state_dict, strict=False)
        print(f"causal init missing keys: {len(m)}, unexpected keys: {len(u)}")

    # 2) Base causal TI2V LoRA, merged on top of the CAUSAL base.
    if args.base_lora_path is not None:
        base_lora_path = args.base_lora_path
        if not os.path.exists(base_lora_path):
            raise FileNotFoundError(f"Base LoRA checkpoint not found: {base_lora_path}")
        print(f"Merging frozen base causal TI2V LoRA from {base_lora_path}")
        merge_lora(
            SimpleNamespace(transformer=transformer3d, text_encoder=text_encoder),
            base_lora_path,
            args.base_lora_multiplier,
            device="cpu",
            dtype=torch.float32,
            transformer_only=True,
        )

    # 3) Action init: overlay ONLY the action (adaLN) modules, keeping the base above intact.
    if args.action_pretrain_path is not None:
        print(f"Loading action (adaLN) modules from {args.action_pretrain_path}")
        action_pretrain = CausalWanActionAdaLNModel.from_pretrained(args.action_pretrain_path).to(weight_dtype)
        # Only the action (adaLN) modules are warm-started here (strict=False below), and those
        # depend on `dim`/`num_heads`/`num_layers`, not on in_dim/out_dim. Checkpoints written
        # before the config fix record out_dim as the flattened head width (out_channels*patch,
        # e.g. 192) rather than the channel count (48); that is cosmetic, so ignore it here.
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
        accelerator.print(
            f"Trainable modules '{args.trainable_modules}'."
        )
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

        new_examples["pixel_values"] = torch.stack(new_examples["pixel_values"])
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
        accelerator.init_trackers("wan2.2-causal-lora")

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_transformer_without_base_lora(save_dir):
        unwrapped_transformer = unwrap_model(transformer3d)
        # Always refresh serialized config from live module shape.
        _sync_model_config_from_weights(unwrapped_transformer)
        if args.base_lora_path is not None:
            # Temporarily subtract merged base LoRA for export, then restore.
            mock_pipeline = SimpleNamespace(transformer=unwrapped_transformer)
            merge_lora(
                mock_pipeline,
                args.base_lora_path,
                -args.base_lora_multiplier,
                device="cpu",
                dtype=torch.float32,
                transformer_only=True,
            )
            unwrapped_transformer.save_pretrained(save_dir)
            merge_lora(
                mock_pipeline,
                args.base_lora_path,
                args.base_lora_multiplier,
                device="cpu",
                dtype=torch.float32,
                transformer_only=True,
            )
        else:
            unwrapped_transformer.save_pretrained(save_dir)

    resume_checkpoint_dir = None
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        ckpt_dir = args.resume_from_checkpoint
        if ckpt_dir.split("/")[-1] == "latest":
            ckpt_dir = os.readlink(ckpt_dir)
        resume_checkpoint_dir = ckpt_dir
        global_step = int(ckpt_dir.split("-")[-1])
        first_epoch = global_step // num_update_steps_per_epoch
        accelerator.print(f"Resuming from checkpoint {ckpt_dir}")
        load_model = CausalWanActionAdaLNModel.from_pretrained(ckpt_dir)
        _assert_same_arch(expected_arch, _extract_arch_signature(load_model), ckpt_dir)
        unwrapped_transformer3d = unwrap_model(transformer3d)
        unwrapped_transformer3d.register_to_config(**load_model.config)
        m, u = unwrapped_transformer3d.load_state_dict(load_model.state_dict(), strict=False)
        accelerator.print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        del load_model
        if args.base_lora_path is not None:
            # Checkpoints are exported with base LoRA subtracted. Re-apply it on resume.
            accelerator.print(f"Re-merging frozen base causal TI2V LoRA from {args.base_lora_path}")
            merge_lora(
                SimpleNamespace(transformer=unwrapped_transformer3d, text_encoder=text_encoder),
                args.base_lora_path,
                args.base_lora_multiplier,
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
            accelerator.print(
                f"[Warning] Optimizer state not found at {optimizer_path}. "
                "Resumed run may have quality/learning-rate discontinuity."
            )
        if os.path.exists(scheduler_path):
            accelerator.print(f"Loading LR scheduler state from {scheduler_path}")
            lr_scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu"))
        else:
            accelerator.print(
                f"[Warning] LR scheduler state not found at {scheduler_path}. "
                "Resumed run may have learning-rate discontinuity."
            )
    else:
        global_step = 0
        first_epoch = 0

    if accelerator.is_main_process and args.validation_steps > 0:
        # Run one eval on the exact starting state (after optional resume load),
        # before entering the training loop.
        log_validation(
            vae,
            text_encoder,
            tokenizer,
            transformer3d,
            config,
            args,
            accelerator,
            weight_dtype,
            global_step,
        )

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
            f"Resume data position: first_epoch={first_epoch}, "
            f"skip_batches_in_first_epoch={resume_batches_to_skip}"
        )

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
                pixel_values = batch["pixel_values"].to(device=accelerator.device, dtype=weight_dtype)
                mouse_actions = batch["mouse_actions"].to(device=accelerator.device, dtype=weight_dtype)
                keyboard_actions = batch["keyboard_actions"].to(device=accelerator.device, dtype=weight_dtype)

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
                effective_i2v = use_i2v and num_frames > 1

                # Match chunk layout with task type for this batch:
                # i2v -> independent_first_frame=True, t2v -> False.
                unwrapped_t = accelerator.unwrap_model(transformer3d)
                set_batch_causal_layout(
                    unwrapped_t,
                    chunk_size=args.chunk_size,
                    use_i2v=effective_i2v,
                    local_attn_size=args.local_attn_size,
                    sink_size=args.sink_size,
                )
                if effective_i2v:
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
                        mouse_actions=mouse_actions,
                        keyboard_actions=keyboard_actions,
                        frame_offset=0,
                        clean_x=latents,
                    )

                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                mse_loss = F.mse_loss(noise_pred.float(), target.float(), reduction='none')
                mse_loss = mse_loss * loss_mask.float()
                if weighting is not None:
                    mse_loss = mse_loss * weighting.float()
                loss = mse_loss.sum() / loss_mask.float().expand_as(mse_loss).sum().clamp(min=1.0)

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()

                # NaN-guard: training can converge fine for a long time and then a single
                # numerical spike (bf16 forward overflow on the long 397-frame sequence)
                # makes the loss non-finite; a backward+step on it permanently poisons the
                # weights with NaN so every later step stays NaN. `avg_loss` is gathered and
                # thus identical on all ranks, so the skip decision stays in sync (no DeepSpeed
                # desync). On a bad batch we skip backward+step entirely -> the NaN gradients
                # never reach the DeepSpeed engine and the weights stay clean.
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
                            f"[nan-guard] non-finite loss ({avg_loss.item()}) at step "
                            f"{global_step}; skipping optimizer.step. text={batch.get('text')}"
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
                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        accelerator_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        save_transformer_without_base_lora(accelerator_save_path)
                        torch.save(optimizer.state_dict(), os.path.join(accelerator_save_path, "optimizer.pt"))
                        torch.save(lr_scheduler.state_dict(), os.path.join(accelerator_save_path, "scheduler.pt"))
                        logger.info(f"Saved state to {accelerator_save_path}")

                if accelerator.is_main_process:
                    if args.validation_steps > 0 and global_step % args.validation_steps == 0:
                        log_validation(
                            vae,
                            text_encoder,
                            tokenizer,
                            transformer3d,
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
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        save_transformer_without_base_lora(save_path)
        torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))
        torch.save(lr_scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))
        logger.info(f"Saved state to {save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
