# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Causal DMD training for Wan-style chunk-wise generation, with an ACTION-AWARE teacher.

The DMD real score (teacher) and fake score (critic) are both the bidirectional action
teacher: vanilla bidirectional Wan2.2 base + teacher LoRA (--teacher_lora_path) + action
adaLN modules (--teacher_action_pretrain_path). Because the teacher is action-conditioned,
the DMD distribution-matching gradient rewards the causal student for FOLLOWING actions --
an action-blind teacher cannot supply that signal, so the distilled student stops
following actions.

If --teacher_action_pretrain_path is omitted, the scores fall back to an action-blind
bidirectional teacher.
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
import shutil
import sys
from types import SimpleNamespace
from collections import OrderedDict

import accelerate
import diffusers
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
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
from microworld.models import (AutoencoderKLWan3_8, WanT5EncoderModel,
                              CausalWanActionAdaLNModel, WanActionAdaLNModel,
                              WanTransformer3DModel)
from microworld.pipeline.pipeline_wan_causal_dmd import WanCausalDMDPipeline
from microworld.utils.lora_utils import create_network, merge_lora
from microworld.utils import lora_utils
from microworld.utils.utils import save_videos_grid
from microworld.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

if is_wandb_available():
    import wandb


def filter_kwargs(cls, kwargs):
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs


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


def _load_transformer_state_dict_file(model, path, tag="Teacher"):
    """Overlay a raw checkpoint file (e.g. CAUSAL model.pt) onto a transformer.

    Used so DMD score networks can be initialized from the causal checkpoint
    instead of vanilla Wan2.2. The causal/bidirectional transformers share the
    same parameter layout, so weights transfer directly; only the attention
    masking differs (the score networks still run bidirectionally)."""
    if path.endswith("safetensors"):
        from safetensors.torch import load_file as _safe_load_file
        raw = _safe_load_file(path)
    else:
        raw = torch.load(path, map_location="cpu")
    sd = _unwrap_transformer_state_dict(raw)
    sd = _normalize_transformer_state_dict_keys(sd)
    model_keys = set(model.state_dict().keys())
    filtered = OrderedDict((k, v) for k, v in sd.items() if k in model_keys)
    if len(filtered) == 0:
        sample = list(sd.keys())[:10]
        raise RuntimeError(f"[DMD {tag}] No keys matched checkpoint {path}. Sample keys: {sample}")
    m, u = model.load_state_dict(filtered, strict=False)
    print(f"[DMD {tag}] Loaded checkpoint into bidirectional score model: {path}")
    print(f"[DMD {tag}] matched: {len(filtered)}/{len(model_keys)}, missing: {len(m)}, unexpected: {len(u)}")
    del raw, sd, filtered


check_min_version("0.18.0.dev0")

logger = get_logger(__name__, log_level="INFO")


@torch.no_grad()
def log_validation(
    one_sample,
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
    mouse_scale=None,
):
    """Validate the current student with the same causal pipeline used by action training."""
    try:
        logger.info("Running causal pipeline validation... ")

        scheduler = FlowUniPCMultistepScheduler(
            **filter_kwargs(FlowUniPCMultistepScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
        )

        transformer_base_path = args.transformer_path
        if transformer_base_path is None or os.path.isfile(transformer_base_path):
            transformer_base_path = os.path.join(
                args.pretrained_model_name_or_path,
                config['transformer_additional_kwargs'].get(
                    'transformer_low_noise_model_subpath',
                    config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer'),
                ),
            )
        transformer3d_val = CausalWanActionAdaLNModel.from_pretrained(
            transformer_base_path,
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        ).to(weight_dtype)
        configure_causal_teacher_forcing(transformer3d_val, args.chunk_size, args.independent_first_frame)
        missing, unexpected = transformer3d_val.load_state_dict(
            accelerator.unwrap_model(transformer3d).state_dict(),
            strict=False,
        )
        if missing or unexpected:
            print(f"[Validation] base load missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

        mock_pipeline = SimpleNamespace(transformer=transformer3d_val)
        merge_lora(mock_pipeline, None, 1, accelerator.device,
                   state_dict=accelerator.unwrap_model(network).state_dict(),
                   transformer_only=True)
        transformer3d_val = mock_pipeline.transformer.to(accelerator.device, dtype=weight_dtype)
        transformer3d_val.eval()

        vae_unwrapped = accelerator.unwrap_model(vae)
        text_encoder_unwrapped = accelerator.unwrap_model(text_encoder)
        pipeline = WanCausalDMDPipeline(
            transformer=transformer3d_val,
            vae=vae_unwrapped,
            tokenizer=tokenizer,
            text_encoder=text_encoder_unwrapped,
            scheduler=scheduler,
        )
        pipeline = pipeline.to(accelerator.device)

        if args.seed is None:
            generator = None
        else:
            generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

        negative_text = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
        validation_num_frames = 81
        pixel_values = one_sample["pixel_values"].to(device=accelerator.device, dtype=weight_dtype)
        original_video = rearrange(pixel_values, "b f c h w -> b c f h w")
        original_video = ((original_video + 1.0) / 2.0).clamp(0, 1)
        input_video = rearrange(pixel_values, "b f c h w -> b c f h w")
        input_video = (input_video[:, :, :1] + 1.0) / 2.0
        input_video = input_video.clamp(0, 1)
        input_video = input_video.repeat(1, 1, validation_num_frames, 1, 1)
        input_video_mask = torch.zeros_like(input_video[:, :1])
        input_video_mask[:, :, 1:] = 255

        text = one_sample["text"][0]
        video_length = int((validation_num_frames - 1) // vae_unwrapped.config.temporal_compression_ratio * vae_unwrapped.config.temporal_compression_ratio) + 1 if validation_num_frames != 1 else 1
        latent_frames = (video_length - 1) // vae_unwrapped.config.temporal_compression_ratio + 1
        mouse_actions = one_sample.get("mouse_actions", None)
        keyboard_actions = one_sample.get("keyboard_actions", None)
        if mouse_actions is not None:
            mouse_actions = mouse_actions.to(device=accelerator.device, dtype=weight_dtype)
        if keyboard_actions is not None:
            keyboard_actions = keyboard_actions.to(device=accelerator.device, dtype=weight_dtype)

        with accelerator.autocast():
            video = pipeline(
                text,
                num_frames=video_length,
                negative_prompt=negative_text,
                height=args.video_sample_size[0],
                width=args.video_sample_size[1],
                generator=generator,
                guidance_scale=1.0,
                num_inference_steps=4,
                max_sequence_length=args.tokenizer_max_length,
                shift=5.0,
                video=input_video[:, :, :video_length],
                mask_video=input_video_mask[:, :, :video_length],
                chunk_size=args.chunk_size,
                mouse_actions=mouse_actions,
                keyboard_actions=keyboard_actions,
                mouse_scale=mouse_scale,
                with_ui=True,
                denoising_step_list=args.denoising_step_list,
                warp_denoising_step=args.warp_denoising_step,
                num_train_timestep=args.num_train_timestep,
                context_noise_level=args.context_noise_level,
            ).videos

            os.makedirs(os.path.join(args.output_dir, "sample"), exist_ok=True)
            original_video = original_video[:, :, :video.shape[2]].to(device=video.device, dtype=video.dtype)
            video = torch.cat([video, original_video], dim=-1)
            sample_path = os.path.join(args.output_dir, f"sample/sample-{global_step}.mp4")
            save_videos_grid(video, sample_path, fps=16)
            with open(os.path.join(args.output_dir, "sample", "prompt.log"), "a") as f:
                f.write(f"{global_step}, latent_frames={latent_frames}, {text}\n")

        del pipeline
        del transformer3d_val
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception as e:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print(f"Streaming eval error with info {e}")
        import traceback
        traceback.print_exc()
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="DMD2 distillation training script.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None, required=True)
    parser.add_argument("--revision", type=str, default=None, required=False)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--train_data_dir", type=str, default=None)
    parser.add_argument("--train_data_meta", type=str, default=None)
    parser.add_argument(
        "--action_pool_root",
        type=str,
        default=None,
        help="Optional directory of action json files (e.g. MW-capsaicin/metadata-detection). "
             "When set, each sample's action trajectory is drawn at random from this pool "
             "instead of from the video's own json; video frames + prompt still come from "
             "--train_data_meta (e.g. DL3DV).",
    )
    parser.add_argument(
        "--mouse_scale",
        type=float,
        nargs="+",
        default=[1.0],
        help="Multiplier applied to the mouse (pitch_delta, yaw_delta) ONLY on the path "
             "that feeds the network (the action conditioning). The UI overlay / cursor "
             "visualization keeps using the raw (unscaled) mouse, so it is unaffected. "
             "Pass one value to scale both axes equally, or two values "
             "'<pitch_scale> <yaw_scale>'. Used e.g. to amplify DL3DV's small mouse motion "
             "up to the capsaicin level when training the capsaicin action model under DMD. "
             "A negative value flips that axis' rotation direction (e.g. when a source's "
             "yaw/pitch sign convention is reversed). Default 1.0 (no change).",
    )
    parser.add_argument("--output_dir", type=str, default="sd-model-finetuned")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use_came", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--vae_mini_batch", type=int, default=32)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--scale_lr", action="store_true", default=False)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--non_ema_revision", type=str, default=None, required=False)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--validation_steps", type=int, default=2000)
    parser.add_argument(
        "--debug_pred_decode_steps",
        type=int,
        default=50,
        help="If > 0, periodically decode pred_image_full every N global steps for debugging.",
    )
    parser.add_argument("--tracker_project_name", type=str, default="text2image-fine-tune")
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--network_alpha", type=int, default=64)
    parser.add_argument("--video_sample_size", nargs=2, type=int, default=None)
    parser.add_argument("--video_sample_stride", type=int, default=4)
    parser.add_argument("--video_sample_n_frames", type=int, default=17)
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
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument(
        "--transformer_path",
        type=str,
        default=None,
        help="CAUSAL init weights (model.pt / .safetensors) for the student base, loaded as a state_dict on top of the Wan2.2 base transformer (causal-forcing initialization).",
    )
    parser.add_argument(
        "--action_pretrain_path",
        type=str,
        default=None,
        help="Optional action checkpoint (CausalWanActionAdaLNModel dir). Only the action modules (action_preprocess + action_adaLN_modulation) are overlaid onto the student base, keeping base+LoRA weights intact.",
    )
    parser.add_argument(
        "--teacher_transformer_path",
        type=str,
        default=None,
        help="Optional pretrained teacher transformer. If it is a directory, it replaces the default teacher transformer loading path; if it is a file, it is loaded as a state_dict.",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Optional game/style LoRA to merge into the frozen student base weights before DMD training.",
    )
    parser.add_argument(
        "--teacher_lora_path",
        type=str,
        default=None,
        help="Optional teacher game/style LoRA to merge into the real score and fake score base weights before DMD training.",
    )
    parser.add_argument(
        "--teacher_action_pretrain_path",
        type=str,
        default=None,
        help="Action checkpoint (WanActionAdaLNModel/CausalWanActionAdaLNModel dir) whose action modules "
             "(action_preprocess + action_adaLN_modulation) are overlaid onto the bidirectional real score "
             "AND fake score, making the DMD teacher/critic action-aware. When set, the score models become "
             "WanActionAdaLNModel and receive the batch's mouse/keyboard actions. This is what lets the "
             "distilled student actually follow actions. Pair it with --teacher_lora_path, i.e. the "
             "teacher LoRA and action modules trained by the two teacher scripts.",
    )
    parser.add_argument("--vae_path", type=str, default=None)
    parser.add_argument('--tokenizer_max_length', type=int, default=512)

    # Chunk-wise generation
    # chunk_size is the chunk size at the latent-frame level; the student writes to the KV cache chunk by chunk.
    parser.add_argument("--chunk_size", type=int, default=3)
    parser.add_argument(
        "--independent_first_frame",
        action="store_true",
        default=True,
        help="Use [1, N, N, ...] chunk layout for causal training/inference.",
    )
    parser.add_argument(
        "--sink_size",
        type=int,
        default=0,
        help="Number of latent frames to commit as clean sink (GT) before rollout. "
             "0 = no sink (default). 1 = first frame from GT data as i2v conditioning. "
             "Only supports sink_size=1 with independent_first_frame=True currently.",
    )

    # DMD distillation arguments
    # These parameters control core DMD behavior: how many steps the student generates,
    # real/fake score CFG, critic learning rate, and the alternating generator/critic update frequency.
    parser.add_argument("--num_denoise_steps", type=int, default=4,
                        help="Number of denoising steps for the student (K). Used only if --denoising_step_list is None.")
    parser.add_argument("--denoising_step_list", type=str, default=None,
                        help="Comma-separated discrete timesteps for the rollout, e.g. '1000,909,717,24'. "
                             "Must match the distill_timesteps used during diffusion forcing pretraining. "
                             "If set, overrides --num_denoise_steps and --shift for the rollout grid.")
    parser.add_argument("--shift", type=float, default=5.0,
                        help="Sigma shift for the timestep schedule. Also used for warping --denoising_step_list when --warp_denoising_step is set.")
    parser.add_argument("--warp_denoising_step", action="store_true", default=True,
                        help="Treat --denoising_step_list as INDICES (1..1000) into the shifted timestep table, "
                             "matching the official Self-Forcing recipe. With shift=5, [1000,750,500,250] warps to "
                             "[1000,937,833,625] and the corresponding sigmas to [1.0,0.938,0.833,0.625].")
    parser.add_argument("--dm_loss_weight", type=float, default=1.0,
                        help="Weight for the distribution matching loss.")
    parser.add_argument("--fake_score_lr", type=float, default=2e-6,
                        help="Learning rate for the fake score (critic) LoRA.")
    parser.add_argument("--fake_score_rank", type=int, default=128,
                        help="LoRA rank for fake score network.")
    parser.add_argument("--dfake_gen_update_ratio", type=int, default=5,
                        help="Train generator every N steps. Critic trains every step.")
    parser.add_argument("--dfake_warmup_steps", type=int, default=0,
                        help="Critic-only warmup: for the first N global steps, ONLY update the critic "
                             "(fake_score) and freeze the generator. Lets the critic learn the student "
                             "distribution before DMD gradients start moving the generator, which prevents "
                             "early collapse. 0 = no warmup.")
    parser.add_argument("--real_guidance_scale", type=float, default=3.0,
                        help="CFG scale for real score in DM loss.")
    parser.add_argument("--fake_guidance_scale", type=float, default=0.0,
                        help="CFG scale for fake score in DM loss. 0 = no CFG.")
    parser.add_argument("--ema_decay", type=float, default=0.99,
                        help="EMA decay rate for student LoRA weights.")
    parser.add_argument("--ema_start_step", type=int, default=200,
                        help="Global step at which to start EMA tracking.")
    parser.add_argument("--timestep_shift", type=float, default=5.0,
                        help="Flow matching timestep shift.")
    parser.add_argument("--num_train_timestep", type=int, default=1000,
                        help="Total number of training timesteps for score networks.")

    # Self-Forcing alignment: KV cache rollout + random exit + ts_schedule
    # random exit means the student rollout stops at a random denoise step,
    # then uses that intermediate x0_pred for DMD score matching.
    parser.add_argument("--rollout_grad_chunks", type=int, default=-1,
                        help="-1 = backprop through all chunks; K = only backprop through last K chunks (frame-dim truncation).")
    parser.add_argument("--same_step_across_blocks", action="store_true", default=True,
                        help="If True, all chunks share the same random exit step (matches Self-Forcing default).")
    parser.add_argument("--ts_schedule", action="store_true", default=True,
                        help="If True, sample DM/critic timesteps within the generator's exit-step interval.")
    parser.add_argument("--ts_schedule_max", action="store_true", default=False,
                        help="If True, also clamp the upper bound of the score timestep range.")
    parser.add_argument("--context_noise_level", type=float, default=0.0,
                        help="Sigma of the noise added before the t=0 cache-update pass between chunks.")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def configure_causal_teacher_forcing(transformer3d, chunk_size, independent_first_frame, sink_size=0):
    transformer3d.chunk_size = chunk_size
    transformer3d.independent_first_frame = independent_first_frame
    for block in transformer3d.blocks:
        block.self_attn.chunk_size = chunk_size
        block.self_attn.independent_first_frame = independent_first_frame
        block.self_attn.sink_size = sink_size


def main():
    args = parse_args()

    now = datetime.now()
    time_str = now.strftime("%Y%m%d-%H%M")
    args.output_dir = os.path.join(args.output_dir, time_str)

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    config = OmegaConf.load(args.config_path)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
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
    torch_rng = torch.Generator(accelerator.device)
    if args.seed is not None:
        torch_rng.manual_seed(args.seed + accelerator.process_index)
        print(f"Init rng with seed {args.seed + accelerator.process_index}. Process_index is {accelerator.process_index}")

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # -------------------------------------------------------------------------
    # 1. Load scheduler / tokenizer / text encoder / VAE.
    # These modules are frozen and used only for text encoding, latent encode/decode, or providing the timestep table.
    # -------------------------------------------------------------------------
    # Load tokenizer and models
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )

    def deepspeed_zero_init_disabled_context_manager():
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []
        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder_path = os.path.join(args.pretrained_model_name_or_path, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder'))
        print(f"[Weight Loading] Loading text encoder from: {text_encoder_path}")
        text_encoder = WanT5EncoderModel.from_pretrained(
            text_encoder_path,
            additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
        text_encoder = text_encoder.eval()

        vae_path = os.path.join(args.pretrained_model_name_or_path, config['vae_kwargs'].get('vae_subpath', 'vae'))
        print(f"[Weight Loading] Loading VAE from: {vae_path}")
        vae = AutoencoderKLWan3_8.from_pretrained(
            vae_path,
            additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
        )
        vae = vae.eval()

    # -------------------------------------------------------------------------
    # 2. Load the student generator's base transformer.
    # It is a causal transformer; only the student LoRA attached on top is trained afterward.
    # -------------------------------------------------------------------------
    # Get Transformer (generator base) — CausalWanActionAdaLNModel (causal attention + action).
    # Student base is built in the exact same order as inference / action training:
    #   base -> CAUSAL init (model.pt) -> base LoRA -> action (adaLN-only) overlay
    # so the student LoRA we distill here plugs onto the same base used at inference time.
    base_subpath = config['transformer_additional_kwargs'].get(
        'transformer_low_noise_model_subpath',
        config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer'),
    )
    base_path = os.path.join(args.pretrained_model_name_or_path, base_subpath)
    print(f"[Weight Loading] Loading generator base (CausalWanActionAdaLNModel) from: {base_path}")

    transformer3d = CausalWanActionAdaLNModel.from_pretrained(
        base_path,
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    ).to(weight_dtype)
    configure_causal_teacher_forcing(transformer3d, args.chunk_size, args.independent_first_frame, args.sink_size)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    transformer3d.requires_grad_(False)

    # 1) CAUSAL init weights (model.pt) -- causal-forcing initialization for the student base.
    if args.transformer_path is not None:
        print(f"[Weight Loading] Loading CAUSAL init weights: {args.transformer_path}")
        if args.transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file as _safe_load_file
            _causal_sd = _safe_load_file(args.transformer_path)
        else:
            _causal_sd = torch.load(args.transformer_path, map_location="cpu")
        _causal_sd = _unwrap_transformer_state_dict(_causal_sd)
        _causal_sd = _normalize_transformer_state_dict_keys(_causal_sd)
        _m, _u = transformer3d.load_state_dict(_causal_sd, strict=False)
        print(f"[Weight Loading] CAUSAL init loaded. missing: {len(_m)}, unexpected: {len(_u)}")
        del _causal_sd

    # 2) Base causal TI2V LoRA merged into the base weights.
    if args.lora_path is not None:
        print(f"[LoRA Loading] Merging base LoRA from: {args.lora_path}")
        mock_pipeline = SimpleNamespace(transformer=transformer3d)
        merge_lora(mock_pipeline, args.lora_path, 1.0, device='cpu', dtype=torch.float32, transformer_only=True)
        print("[LoRA Loading] Base LoRA merged into student base weights.")

    # 3) Overlay ONLY the action modules (adaLN), keeping the base above intact.
    if args.action_pretrain_path is not None:
        print(f"[Weight Loading] Overlaying action (adaLN) modules from: {args.action_pretrain_path}")
        action_model = CausalWanActionAdaLNModel.from_pretrained(
            args.action_pretrain_path,
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        ).to(weight_dtype)
        _full_sd = action_model.state_dict()
        _action_sd = {
            k: v for k, v in _full_sd.items()
            if k.startswith("action_preprocess") or "action_adaLN_modulation" in k
        }
        _m, _u = transformer3d.load_state_dict(_action_sd, strict=False)
        print(f"[Weight Loading] Overlaid {len(_action_sd)} action params; unexpected: {len(_u)}")
        del action_model, _full_sd, _action_sd

    # Student LoRA (trainable) — the DMD few-step causal generator parameters.
    print(f"[LoRA Loading] Creating trainable student LoRA network (rank={args.rank}, alpha={args.network_alpha})")
    if "CausalWanActionAdaLNModel" not in lora_utils.LoRANetwork.TRANSFORMER_TARGET_REPLACE_MODULE:
        lora_utils.LoRANetwork.TRANSFORMER_TARGET_REPLACE_MODULE.append("CausalWanActionAdaLNModel")

    # Exclude the action modules (action_preprocess + action_adaLN_modulation) from the
    # student LoRA so the pretrained action->motion control interface stays FROZEN. The DMD
    # objective is action-blind (teacher/fake-score have no action modules), so letting it
    # retrain the action path would attenuate the action response magnitude.
    network = create_network(
        1.0, args.rank, args.network_alpha,
        text_encoder, transformer3d, neuron_dropout=None,
        skip_name="action",
    )
    network.apply_to(text_encoder, transformer3d, False, True)

    # Resume trained LoRA weights BEFORE accelerator.prepare so DeepSpeed's
    # NOTE: load student LoRA here before accelerator.prepare so DeepSpeed FP32 master
    # weights are not left at random init, which would cause the first optimizer.step to overwrite loaded weights.
    # FP32 master copies get initialized from the loaded values rather than
    # the fresh-init zeros. Loading after prepare leaves master weights at
    # fresh init, and the first optimizer.step() then casts master->BF16 and
    # wipes out the loaded weights.
    if args.resume_from_checkpoint:
        _ckpt_dir = args.resume_from_checkpoint
        if _ckpt_dir.split("/")[-1] == "latest":
            _ckpt_dir = os.readlink(_ckpt_dir)
        _lora_raw_path = os.path.join(_ckpt_dir, "lora_raw.safetensors")
        _lora_main_path = os.path.join(_ckpt_dir, "lora_diffusion_pytorch_model.safetensors")
        _lora_ckpt_path = _lora_raw_path if os.path.exists(_lora_raw_path) else _lora_main_path
        if os.path.exists(_lora_ckpt_path):
            print(f"[LoRA Loading] Pre-prepare LoRA resume from: {_lora_ckpt_path}")
            from safetensors.torch import load_file as _safe_load_file
            _state_dict = _safe_load_file(_lora_ckpt_path)
            _m, _u = network.load_state_dict(_state_dict, strict=False)
            print(f"[LoRA Loading] Pre-prepare LoRA loaded. missing keys: {len(_m)}, unexpected keys: {len(_u)}")
            if len(_m) > 0:
                print(f"[LoRA Loading] missing (sample): {_m[:5]}")
            if len(_u) > 0:
                print(f"[LoRA Loading] unexpected (sample): {_u[:5]}")
        else:
            print(f"[LoRA Loading] No LoRA file found at resume dir {_ckpt_dir}; skipping pre-prepare load.")

    if args.vae_path is not None:
        print(f"[Weight Loading] Loading custom VAE weights from: {args.vae_path}")
        if args.vae_path.endswith("safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(args.vae_path)
        else:
            state_dict = torch.load(args.vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        m, u = vae.load_state_dict(state_dict, strict=False)
        assert len(u) == 0

    if args.gradient_checkpointing:
        transformer3d.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # -------------------------------------------------------------------------
    # 4. Student generator optimizer.
    # The optimizer updates only student LoRA; base transformer / VAE / text encoder remain frozen.
    # -------------------------------------------------------------------------
    optimizer_cls = torch.optim.AdamW

    trainable_params = list(filter(lambda p: p.requires_grad, network.parameters()))
    trainable_params_optim = network.prepare_optimizer_params(args.learning_rate / 2, args.learning_rate, args.learning_rate)

    optimizer = optimizer_cls(
        trainable_params_optim,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # -------------------------------------------------------------------------
    # 5. Dataset.
    # The original script uses use_action=True and loads mouse_actions / keyboard_actions.
    # For TI2V DMD, this is typically switched to no action and first-frame conditioning logic is added.
    # -------------------------------------------------------------------------
    # Dataset
    train_dataset = VideoGameDataset(
        ann_path=args.train_data_meta,
        data_root=args.train_data_dir,
        video_sample_size=args.video_sample_size,
        video_sample_stride=args.video_sample_stride,
        video_sample_n_frames=args.video_sample_n_frames,
        use_action=True,
        action_pool_root=args.action_pool_root,
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

    network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        network, optimizer, train_dataloader, lr_scheduler
    )

    # Initialize trackers so accelerator.log(...) actually writes to tensorboard
    # (or whatever backend --report_to specifies). Without this call, every
    # accelerator.log() is a silent no-op and the events file stays empty.
    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name)

    vae.to(accelerator.device, dtype=weight_dtype)
    transformer3d.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running DMD2 training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    resume_checkpoint_path = None

    # Resume checkpoint
    if args.resume_from_checkpoint:
        ckpt_dir = args.resume_from_checkpoint
        if ckpt_dir.split("/")[-1] == "latest":
            ckpt_dir = os.readlink(ckpt_dir)
        resume_checkpoint_path = ckpt_dir
        global_step = int(ckpt_dir.split("-")[-1])
        initial_global_step = global_step
        first_epoch = global_step // num_update_steps_per_epoch
        print(f"[Checkpoint Resume] Resuming from checkpoint: {ckpt_dir}")
        print(f"[Checkpoint Resume] Resuming from global_step: {global_step}")

        is_dmd_checkpoint = os.path.exists(os.path.join(ckpt_dir, "fake_score_lora.safetensors"))
        if is_dmd_checkpoint:
            for fname, target, label in [
                ("optimizer.pt", optimizer, "optimizer"),
                ("scheduler.pt", lr_scheduler, "LR scheduler"),
            ]:
                fpath = os.path.join(ckpt_dir, fname)
                if os.path.exists(fpath):
                    try:
                        state = torch.load(fpath, map_location=accelerator.device)
                        target.load_state_dict(state)
                        print(f"[Checkpoint Resume] {label} loaded from {fname}")
                    except Exception as e:
                        print(f"[Checkpoint Resume] Failed to load {label} from {fname}: {e}")
    else:
        initial_global_step = 0

    # =========================================================================
    # DMD-specific setup: Real Score, Fake Score, Negative Prompt, EMA
    # From here begins DMD-specific setup: build real score, fake score,
    # pre-encode negative prompt, and prepare student EMA.
    # =========================================================================

    # --- Real score (frozen bidirectional model, Wan2.1 original + game LoRA) ---
    # real score acts as the frozen teacher. It provides the denoising direction of the "real data distribution".
    # In DMD the generator is not supervised directly with GT video; instead, student-generated samples are re-noised
    # and passed to real score to estimate "where to denoise if this sample came from the real distribution".
    # Following Causal-Forcing: both real/fake score use bidirectional WanTransformer3DModel,
    # i.e. score evaluation can see the full video without causal/chunk attention constraints.
    # The student remains a causal generator; bidirectional scores only provide training gradients.
    # Action-aware teacher variant: real/fake score are the bidirectional teacher, i.e.
    # vanilla bidirectional Wan2.2 base + teacher LoRA + action (adaLN) modules.
    # Because the real score is now action-conditioned,
    # the DMD distribution-matching gradient rewards the student for FOLLOWING the actions, which
    # a plain action-blind teacher (WanTransformer3DModel) can't do. When
    # --teacher_action_pretrain_path is unset the scores fall back to the original action-blind
    # bidirectional teacher.
    # Teacher resolution: a directory is used as the from_pretrained source; a file (e.g. CAUSAL
    # model.pt) is overlaid as a state_dict onto the base architecture.
    use_teacher_action = args.teacher_action_pretrain_path is not None
    ScoreModelClass = WanActionAdaLNModel if use_teacher_action else WanTransformer3DModel
    teacher_is_file = args.teacher_transformer_path is not None and os.path.isfile(args.teacher_transformer_path)
    teacher_dir = base_path
    if args.teacher_transformer_path is not None and os.path.isdir(args.teacher_transformer_path):
        teacher_dir = args.teacher_transformer_path

    def _overlay_teacher_action(model, tag):
        # Overlay ONLY the action modules (action_preprocess + per-block action_adaLN_modulation)
        # from the action checkpoint, keeping the (base + teacher LoRA) weights intact. Shares the
        # exact param layout across the causal/bidirectional action classes so a causal-trained
        # action checkpoint transfers directly.
        print(f"[DMD {tag}] Overlaying teacher action modules from: {args.teacher_action_pretrain_path}")
        action_model = WanActionAdaLNModel.from_pretrained(
            args.teacher_action_pretrain_path,
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
        full_sd = action_model.state_dict()
        action_sd = {
            k: v for k, v in full_sd.items()
            if k.startswith("action_preprocess") or "action_adaLN_modulation" in k
        }
        m, u = model.load_state_dict(action_sd, strict=False)
        print(f"[DMD {tag}] loaded {len(action_sd)} action params; unexpected keys: {len(u)}")
        del action_model, full_sd

    print(f"[DMD Real Score] Creating frozen real score model (bidirectional, action={use_teacher_action})...")
    real_score_transformer3d = ScoreModelClass.from_pretrained(
        teacher_dir,
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        # Materialize action modules (absent from the base checkpoint) with real storage so the
        # later in-place action load_state_dict is not a no-op on meta tensors.
        low_cpu_mem_usage=not use_teacher_action,
    ).to(weight_dtype)
    if teacher_is_file:
        _load_transformer_state_dict_file(real_score_transformer3d, args.teacher_transformer_path, tag="Real Score")
    real_score_transformer3d.requires_grad_(False)
    real_score_transformer3d.eval()
    if args.gradient_checkpointing:
        real_score_transformer3d.enable_gradient_checkpointing()
    real_score_transformer3d.to(accelerator.device)
    print(f"[DMD Real Score] Created. Parameters: {sum(p.numel() for p in real_score_transformer3d.parameters()) / 1e6:.1f}M")
    if args.teacher_lora_path is not None:
        print(f"[LoRA Loading] Merging game LoRA from: {args.teacher_lora_path}")
        mock_pipeline = SimpleNamespace(transformer=real_score_transformer3d)
        merge_lora(mock_pipeline, args.teacher_lora_path, 1.0, device='cpu', dtype=torch.float32, transformer_only=True)
        print("[LoRA Loading] Teacher LoRA merged into real score.")
    if use_teacher_action:
        _overlay_teacher_action(real_score_transformer3d, "Real Score")
        real_score_transformer3d.requires_grad_(False)

    # --- Fake score (critic: bidirectional Wan2.1 + game LoRA + trainable critic LoRA) ---
    # fake score is the critic; it trains a LoRA to fit the score of the student's current generation distribution.
    # Intuition:
    #   - real_score: "What is the denoising direction at x_t for the real video distribution?"
    #   - fake_score: "What is the denoising direction at x_t for the student's current generation distribution?"
    # On generator updates, fake_score - real_score is used as the distribution-matching gradient;
    # on critic updates, standard flow-matching MSE lets fake_score track the student's latest distribution.
    print(f"[DMD Fake Score] Creating fake score model (bidirectional + critic LoRA, action={use_teacher_action})...")
    fake_score_transformer3d = ScoreModelClass.from_pretrained(
        teacher_dir,
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=not use_teacher_action,
    ).to(weight_dtype)
    if teacher_is_file:
        _load_transformer_state_dict_file(fake_score_transformer3d, args.teacher_transformer_path, tag="Fake Score")
    fake_score_transformer3d.requires_grad_(False)
    # Merge teacher LoRA and overlay action BEFORE creating the critic LoRA, so the critic's
    # trainable LoRA sits on top of the same (base + teacher LoRA + action) weights the real
    # score uses. The critic must see the exact same conditioning as the teacher.
    if args.teacher_lora_path is not None:
        print(f"[LoRA Loading] Merging game LoRA from: {args.teacher_lora_path}")
        mock_pipeline = SimpleNamespace(transformer=fake_score_transformer3d)
        merge_lora(mock_pipeline, args.teacher_lora_path, 1.0, device='cpu', dtype=torch.float32, transformer_only=True)
        print("[LoRA Loading] Teacher LoRA merged into fake score.")
    if use_teacher_action:
        _overlay_teacher_action(fake_score_transformer3d, "Fake Score")
        fake_score_transformer3d.requires_grad_(False)

    fake_score_network = create_network(
        1.0, args.fake_score_rank, args.fake_score_rank,
        text_encoder, fake_score_transformer3d, neuron_dropout=None,
    )
    fake_score_network.apply_to(text_encoder, fake_score_transformer3d, False, True)
    if args.gradient_checkpointing:
        fake_score_transformer3d.enable_gradient_checkpointing()
    fake_score_transformer3d.to(accelerator.device, dtype=weight_dtype)
    fake_score_network = fake_score_network.to(accelerator.device)
    print(f"[DMD Fake Score] Critic LoRA params: {sum(p.numel() for p in fake_score_network.parameters()) / 1e6:.1f}M")
    # Fake score optimizer
    # NOTE: fake_score_network is the critic LoRA, separate from student LoRA parameters.
    # It has its own optimizer and learning rate fake_score_lr; the critic updates every step in the training loop,
    # while the student generator updates only on steps controlled by dfake_gen_update_ratio.
    fake_score_optimizer = torch.optim.AdamW(
        fake_score_network.parameters(),
        lr=args.fake_score_lr,
        betas=(0.0, 0.999),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    print(f"[DMD Fake Score] Optimizer created (lr={args.fake_score_lr})")

    # Load fake score checkpoint if resuming from DMD2 checkpoint
    if resume_checkpoint_path is not None:
        ckpt_dir_dmd = resume_checkpoint_path
        fs_lora_path = os.path.join(ckpt_dir_dmd, "fake_score_lora.safetensors")
        fs_opt_path = os.path.join(ckpt_dir_dmd, "fake_score_optimizer.pt")
        if os.path.exists(fs_lora_path):
            print(f"[DMD Fake Score] Loading LoRA weights from {fs_lora_path}")
            from safetensors.torch import load_file as safe_load_file
            fs_state_dict = safe_load_file(fs_lora_path)
            fake_score_network.load_state_dict(fs_state_dict, strict=False)
            print("[DMD Fake Score] LoRA loaded.")
        if os.path.exists(fs_opt_path):
            print(f"[DMD Fake Score] Loading optimizer state from {fs_opt_path}")
            fs_opt_state = torch.load(fs_opt_path, map_location=accelerator.device)
            fake_score_optimizer.load_state_dict(fs_opt_state)
            print("[DMD Fake Score] Optimizer state loaded.")

    # --- Pre-encode negative prompt for real score CFG ---
    # real score uses CFG by default, so pre-encode negative prompt to avoid re-encoding every step.
    # CFG in DMD is used during score estimation, not as extra guidance at sampling time.
    # Higher real_score CFG pushes the "real distribution direction" the generator receives toward text conditioning,
    # equivalent to distilling guidance into student weights.
    NEGATIVE_PROMPT = (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
        "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
        "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
        "杂乱的背景，三条腿，背景人很多，倒着走"
    )
    print("[DMD] Pre-encoding negative prompt for CFG...")
    with torch.no_grad():
        _neg_ids = tokenizer(
            [NEGATIVE_PROMPT],
            padding="max_length", max_length=args.tokenizer_max_length,
            truncation=True, add_special_tokens=True, return_tensors="pt",
        )
        _neg_input_ids = _neg_ids.input_ids.to(accelerator.device)
        _neg_mask = _neg_ids.attention_mask.to(accelerator.device)
        _neg_seq_lens = _neg_mask.gt(0).sum(dim=1).long()
        _neg_out = text_encoder(_neg_input_ids, attention_mask=_neg_mask)[0]
        neg_prompt_embed_single = _neg_out[0, :_neg_seq_lens[0]].to(weight_dtype)
    print(f"[DMD] Negative prompt encoded. Shape: {neg_prompt_embed_single.shape}")

    # --- EMA for student LoRA ---
    # EMA stores a smoother student LoRA. Checkpoints can save the EMA version as the main LoRA,
    # while raw LoRA is used to resume training.
    # EMA tracks only student generator LoRA, not fake_score critic.
    # Reason: the student is what gets deployed/inferred; the critic is only an auxiliary score model during training.
    ema_shadow = None
    if initial_global_step >= args.ema_start_step:
        ema_shadow = {name: param.data.clone() for name, param in network.named_parameters() if param.requires_grad}
        if resume_checkpoint_path is not None:
            ema_path = os.path.join(resume_checkpoint_path, "ema_lora.pt")
            if os.path.exists(ema_path):
                ema_state = torch.load(ema_path, map_location=accelerator.device)
                ema_shadow.update(ema_state)
                print(f"[EMA] Loaded from {ema_path}")
        print(f"[EMA] Initialized (step {initial_global_step} >= ema_start {args.ema_start_step}).")
    else:
        print(f"[EMA] Will start tracking at step {args.ema_start_step} (decay={args.ema_decay}).")

    def update_ema(ema_dict, net, decay):
        with torch.no_grad():
            for name, param in net.named_parameters():
                if param.requires_grad and name in ema_dict:
                    ema_dict[name].mul_(decay).add_(param.data, alpha=1.0 - decay)

    def apply_ema(ema_dict, net):
        backup = {}
        for name, param in net.named_parameters():
            if param.requires_grad and name in ema_dict:
                backup[name] = param.data.clone()
                param.data.copy_(ema_dict[name])
        return backup

    def restore_from_ema(backup, net):
        for name, param in net.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])

    def save_model(ckpt_file, unwrapped_nw):
        os.makedirs(args.output_dir, exist_ok=True)
        accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
        unwrapped_nw.save_weights(ckpt_file, weight_dtype, None)

    def save_training_checkpoint(save_path):
        os.makedirs(save_path, exist_ok=True)
        wrapped_network = network
        unwrapped_network = accelerator.unwrap_model(network)

        save_model(os.path.join(save_path, "lora_raw.safetensors"), unwrapped_network)
        if ema_shadow is not None:
            ema_backup = apply_ema(ema_shadow, wrapped_network)
            save_model(os.path.join(save_path, "lora_diffusion_pytorch_model.safetensors"), unwrapped_network)
            restore_from_ema(ema_backup, wrapped_network)
            torch.save(ema_shadow, os.path.join(save_path, "ema_lora.pt"))
        else:
            save_model(os.path.join(save_path, "lora_diffusion_pytorch_model.safetensors"), unwrapped_network)

        save_model(os.path.join(save_path, "fake_score_lora.safetensors"), fake_score_network)
        torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))
        torch.save(lr_scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))
        torch.save(fake_score_optimizer.state_dict(), os.path.join(save_path, "fake_score_optimizer.pt"))

    # -------------------------------------------------------------------------
    # 6. Pre-compute student few-step sigma / timestep table.
    # denoising_step_list reproduces Self-Forcing discrete distill timesteps.
    # -------------------------------------------------------------------------
    # student rollout uses few-step denoising, not full 1000-step diffusion.
    # Training must expose the student to the same timestep sequence as pretrain/inference:
    #   - if denoising_step_list is provided, use those discrete steps explicitly;
    #   - if warp_denoising_step is enabled, map "raw indices" to shifted scheduler timesteps;
    #   - otherwise use K-step timesteps from FlowUniPCMultistepScheduler.
    # These sigma/timestep values determine each chunk's initial noise level and where Self-Forcing
    # backward simulation re-noises during rollout.
    # Pre-compute sigma schedules
    student_scheduler = FlowUniPCMultistepScheduler(
        **filter_kwargs(FlowUniPCMultistepScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )
    student_scheduler.set_timesteps(args.num_denoise_steps, device=accelerator.device, shift=args.shift)

    # Self-Forcing warp table: full 1000-step shifted FlowMatchEuler timesteps,
    # so denoising_step_list values can be looked up as INDICES (1..1000).
    if args.warp_denoising_step:
        # shift is already baked in via scheduler_kwargs (config.scheduler_kwargs.shift=5.0)
        _warp_scheduler = FlowMatchEulerDiscreteScheduler(
            **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
        )
        _warp_scheduler.set_timesteps(num_inference_steps=args.num_train_timestep, device='cpu')
        warp_table = torch.cat([_warp_scheduler.timesteps.cpu().float(), torch.tensor([0.0])])
    else:
        warp_table = None

    if args.denoising_step_list is not None:
        _ts_print = [int(float(x)) for x in args.denoising_step_list.split(",")]
        if args.warp_denoising_step:
            _warped = warp_table[args.num_train_timestep - torch.tensor(_ts_print, dtype=torch.long)].tolist()
            print(f"[DMD] Student: K={len(_ts_print)} steps, raw_timesteps={_ts_print}, warped_timesteps={_warped} (shift={args.shift})")
        else:
            print(f"[DMD] Student: K={len(_ts_print)} steps, timesteps={_ts_print} (from --denoising_step_list, must match DF distill_timesteps)")
    else:
        print(f"[DMD] Student: K={args.num_denoise_steps} steps, timesteps={student_scheduler.timesteps.tolist()} (from scheduler, shift={args.shift})")
    print(f"[DMD] rollout_grad_chunks={args.rollout_grad_chunks}, same_step_across_blocks={args.same_step_across_blocks}")
    print(f"[DMD] DM loss weight: {args.dm_loss_weight}, dfake_gen_update_ratio: {args.dfake_gen_update_ratio}")
    print(f"[DMD] ts_schedule={args.ts_schedule}, ts_schedule_max={args.ts_schedule_max}, context_noise_level={args.context_noise_level}")

    # DMD timestep range
    min_score_step = int(0.02 * args.num_train_timestep)
    max_score_step = int(0.98 * args.num_train_timestep)

    # Per-axis mouse scale applied ONLY to the network-bound action tensor (not the UI
    # overlay). mouse_actions are ordered [pitch_delta, yaw_delta] in the last dim.
    _ms = list(args.mouse_scale)
    _pitch_scale = float(_ms[0])
    _yaw_scale = float(_ms[1]) if len(_ms) > 1 else float(_ms[0])
    mouse_scale_vec = torch.tensor([_pitch_scale, _yaw_scale], dtype=weight_dtype)
    apply_mouse_scale = (_pitch_scale != 1.0) or (_yaw_scale != 1.0)
    if apply_mouse_scale:
        print(f"[DMD] mouse_scale (network input only): pitch={_pitch_scale}, yaw={_yaw_scale}")

    # Baseline eval runs once on the first batch inside the loop (see below).
    # NOTE: do NOT create a throwaway `iter(train_dataloader)` here — under
    # accelerate+DeepSpeed the prepared dataloader is single-pass/engine-bound,
    # so consuming a separate iterator leaves the real training loop empty.
    ran_init_eval = bool(args.resume_from_checkpoint)

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        train_gen_loss = 0.0
        train_critic_loss = 0.0
        batch_sampler.sampler.generator = torch.Generator().manual_seed((args.seed or 0) + epoch)
        for step, batch in enumerate(train_dataloader):
            mouse_actions = batch.get("mouse_actions", None)
            keyboard_actions = batch.get("keyboard_actions", None)
            if mouse_actions is not None:
                mouse_actions = mouse_actions.to(weight_dtype)
                if apply_mouse_scale:
                    mouse_actions = mouse_actions * mouse_scale_vec.to(mouse_actions.device)
            if keyboard_actions is not None:
                keyboard_actions = keyboard_actions.to(weight_dtype)

            # Actions fed to the action-aware DMD teacher/critic (real & fake score). Empty when
            # the teacher is action-blind, so the WanTransformer3DModel forward signature (which
            # takes no action args) stays unchanged in that fallback path.
            score_action_kwargs = (
                {"mouse_actions": mouse_actions, "keyboard_actions": keyboard_actions}
                if use_teacher_action else {}
            )

            if epoch == first_epoch and step == 0:
                pixel_values, texts = batch['pixel_values'].cpu(), batch['text']
                pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                os.makedirs(os.path.join(args.output_dir, "sanity_check"), exist_ok=True)
                for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                    pixel_value = pixel_value[None, ...]
                    gif_name = '-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_step}-{idx}'
                    save_videos_grid(pixel_value, f"{args.output_dir}/sanity_check/{gif_name[:10]}.gif", rescale=True)

            # Baseline eval BEFORE any DMD update: reference point for distillation.
            # Triggered once on the first real batch (reuses `batch`, no extra iterator).
            if not ran_init_eval:
                ran_init_eval = True
                if accelerator.is_main_process and args.validation_steps > 0:
                    init_sample = {
                        "pixel_values": batch["pixel_values"][0:1],
                        "text": batch["text"][0:1],
                        "idx": batch["idx"][0:1],
                        "mouse_actions": batch["mouse_actions"][0:1],
                        "keyboard_actions": batch["keyboard_actions"][0:1],
                    }
                    print("[Init Eval] Running baseline validation before training...")
                    log_validation(
                        init_sample, vae, text_encoder, tokenizer,
                        transformer3d, network, config, args,
                        accelerator, weight_dtype, global_step,
                        mouse_scale=args.mouse_scale,
                    )
                accelerator.wait_for_everyone()

            with accelerator.accumulate(transformer3d):
                pixel_values = batch["pixel_values"].to(weight_dtype)

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
                        return torch.cat(new_pixel_values, dim=0)
                    latents = _batch_encode_vae(pixel_values)

                with torch.no_grad():
                    prompt_ids = tokenizer(
                        batch['text'],
                        padding="max_length", max_length=args.tokenizer_max_length,
                        truncation=True, add_special_tokens=True, return_tensors="pt"
                    )
                    text_input_ids = prompt_ids.input_ids
                    prompt_attention_mask = prompt_ids.attention_mask
                    seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
                    prompt_embeds = text_encoder(text_input_ids.to(latents.device), attention_mask=prompt_attention_mask.to(latents.device))[0]
                    # T5 returns fp32 hidden states (fp32 layernorm) even with bf16 weights;
                    # cast to weight_dtype so the bidirectional real/fake score (no internal
                    # context cast) and the student both receive matching-dtype context.
                    prompt_embeds = [u[:v].to(weight_dtype) for u, v in zip(prompt_embeds, seq_lens)]

                bsz, channel, num_frames, height, width = latents.size()
                noise = torch.randn(latents.size(), device=latents.device, generator=torch_rng, dtype=weight_dtype)

                patch_h = accelerator.unwrap_model(transformer3d).config.patch_size[1]
                patch_w = accelerator.unwrap_model(transformer3d).config.patch_size[2]
                HW_patched = (height * width) // (patch_h * patch_w)

                # Determine if this is a generator step or critic-only step
                # critic updates every step; generator updates once every dfake_gen_update_ratio steps.
                # This stabilizes DMD:
                #   - fake_score/critic must continuously track the student's current generation distribution;
                #   - if the generator updates every step following an under-trained critic, training oscillates;
                #   - therefore most steps train only the critic; periodic steps apply DM loss to update the student.
                # critic-only warmup: for the first dfake_warmup_steps, train only the critic and freeze the generator,
                # letting fake_score learn the student distribution first to avoid early DMD gradients destabilizing the generator.
                TRAIN_GENERATOR = (
                    global_step >= args.dfake_warmup_steps
                    and global_step % args.dfake_gen_update_ratio == 0
                )

                # Align num_frames across ranks so num_chunks matches in distributed training.
                if accelerator.num_processes > 1:
                    nf_t = torch.tensor([num_frames], device=latents.device, dtype=torch.long)
                    torch.distributed.all_reduce(nf_t, op=torch.distributed.ReduceOp.MIN)
                    nf_sync = int(nf_t.item())
                    if nf_sync != num_frames:
                        num_frames = nf_sync
                        latents = latents[:, :, :num_frames].contiguous()

                chunk_size = args.chunk_size
                sink_size = args.sink_size
                if sink_size > 0 and not args.independent_first_frame:
                    raise ValueError("sink_size > 0 requires --independent_first_frame")

                # rollout_start_frame: index of the first latent frame to generate after sink frames.
                # With sink: frame 0 is the GT sink (1 frame), rollout starts at frame 1;
                # without sink: rollout starts at frame 0 (original behavior).
                rollout_start_frame = sink_size  # 0 or 1
                generated_frames = num_frames - rollout_start_frame
                generated_chunks = math.ceil(generated_frames / chunk_size)
                num_chunks = generated_chunks  # total chunks = number of chunks to generate

                total_gen_loss = torch.tensor(0.0, device=latents.device)
                total_critic_loss = torch.tensor(0.0, device=latents.device)
                total_critic_loss_val = 0.0

                neg_prompt_embeds = [neg_prompt_embed_single.clone() for _ in range(bsz)]

                # ===== Allocate rolling KV cache for the causal generator =====
                # The student generates causally/chunk-wise, so training explicitly maintains a KV cache.
                # Each chunk's clean commit pass writes the generated x0 into the cache for subsequent chunks.
                # This is the main difference from ordinary bidirectional score training:
                #   - student rollout must simulate streaming causal conditions at inference;
                #   - the current chunk can only depend on history chunks committed to the KV cache;
                #   - real/fake score later evaluate on the concatenated pred_image_full,
                #     but they do not change the student's causal generation path.
                # AdaLN action conditioning does not maintain its own KV cache: action features are
                # deterministic per chunk and are injected into each transformer block's modulation.
                unwrapped_t = accelerator.unwrap_model(transformer3d)
                num_layers_cfg = unwrapped_t.num_layers
                num_heads_cfg = unwrapped_t.num_heads
                head_dim_cfg = unwrapped_t.dim // num_heads_cfg
                frame_seqlen = HW_patched
                max_seq_len_rollout = num_frames * frame_seqlen

                def _create_kv_caches(n_layers):
                    return [{
                        "k": torch.zeros(bsz, max_seq_len_rollout, num_heads_cfg, head_dim_cfg,
                                         device=latents.device, dtype=weight_dtype),
                        "v": torch.zeros(bsz, max_seq_len_rollout, num_heads_cfg, head_dim_cfg,
                                         device=latents.device, dtype=weight_dtype),
                        "global_end_index": torch.tensor(0, dtype=torch.long, device=latents.device),
                        "local_end_index": torch.tensor(0, dtype=torch.long, device=latents.device),
                    } for _ in range(n_layers)]

                def _snapshot_caches(caches):
                    return [(c["global_end_index"].clone(), c["local_end_index"].clone()) for c in caches]

                def _restore_caches(caches, snaps):
                    for c, (g, l) in zip(caches, snaps):
                        c["global_end_index"].copy_(g)
                        c["local_end_index"].copy_(l)

                kv_caches = _create_kv_caches(num_layers_cfg)

                # ===== Random exit step (Self-Forcing style) =====
                # Randomly pick an exit point among few-step denoising to avoid the student learning only the final step,
                # while letting DMD score constrain the student across different noise ranges.
                # Example: with K=4, if exit_step=1, the student rollouts only to the 2nd denoise step,
                # then uses the estimated x0_pred at that point for DMD score loss.
                # This way the student is pulled toward the target distribution at various intermediate states,
                # rather than receiving supervision only on the final x0.
                # Use the same discrete timesteps that the diffusion forcing pretrain saw,
                # otherwise the student is fed OOD noise levels.
                if args.denoising_step_list is not None:
                    raw_ts = torch.tensor(
                        [int(float(x)) for x in args.denoising_step_list.split(",")],
                        dtype=torch.long,
                    )
                    if args.warp_denoising_step:
                        # Self-Forcing: treat raw_ts as INDICES into the shifted timestep table
                        warped = warp_table[args.num_train_timestep - raw_ts]
                        denoise_timesteps = warped.long().to(latents.device)
                        denoise_sigmas = (warped.float() / float(args.num_train_timestep)).to(latents.device)
                    else:
                        denoise_timesteps = raw_ts.to(latents.device)
                        denoise_sigmas = denoise_timesteps.float() / 1000.0
                else:
                    student_scheduler.set_timesteps(args.num_denoise_steps, device=latents.device, shift=args.shift)
                    denoise_timesteps = student_scheduler.timesteps.clone()
                    denoise_sigmas = student_scheduler.sigmas.clone() if hasattr(student_scheduler, 'sigmas') else None
                num_denoise_steps_actual = len(denoise_timesteps)

                if args.same_step_across_blocks:
                    exit_t = torch.randint(0, num_denoise_steps_actual, (1,), device=latents.device)
                else:
                    # Sample exit step only for actually generated chunks (excluding sink chunk)
                    exit_t = torch.randint(0, num_denoise_steps_actual, (generated_chunks,), device=latents.device)
                if accelerator.num_processes > 1:
                    torch.distributed.broadcast(exit_t, src=0)
                if args.same_step_across_blocks:
                    exit_steps = [exit_t.item()] * generated_chunks
                else:
                    exit_steps = exit_t.tolist()

                # Track exit timestep range for ts_schedule (uses chunk-0's exit step;
                # in non-same-step mode this is just an approximation to the score window)
                # ts_schedule goal: score-network re-noise timesteps should not be far from the student's
                # exit point. E.g. if the student rollouts from t=750 to t=500 then exits,
                # real/fake score also preferentially evaluate in that noise range, reducing OOD score constraints.
                exit_step_used = exit_steps[0]
                if exit_step_used >= num_denoise_steps_actual - 1:
                    denoised_timestep_to = 0
                    denoised_timestep_from = int(denoise_timesteps[exit_step_used].item())
                else:
                    denoised_timestep_to = int(denoise_timesteps[exit_step_used + 1].item())
                    denoised_timestep_from = int(denoise_timesteps[exit_step_used].item())

                # Frame-dim grad truncation: only backprop through last K chunks
                # Keep gradients only on the last K chunks to reduce memory for long-video rollout.
                # Historical chunks still generate and write to KV cache but can be detached;
                # subsequent chunks retain full conditioning while the backward graph does not span the entire video.
                grad_chunks = args.rollout_grad_chunks
                # first_grad_chunk is the local index within the generated chunk sequence
                first_grad_chunk = 0 if grad_chunks < 0 else max(0, generated_chunks - grad_chunks)

                debug_decode_now = (
                    args.debug_pred_decode_steps > 0
                    and accelerator.is_main_process
                    and global_step % args.debug_pred_decode_steps == 0
                )
                chunk_outputs = []  # collect per-chunk x0 predictions
                # chunk_outputs stores each chunk's x0_pred. These are concatenated along the frame dim into
                # pred_image_full for whole-video DMD score evaluation.

                with accelerator.autocast():
                    # ===== Sink commit pass =====
                    # If sink_size > 0, forward the GT first frame (t=0, clean) once into the KV cache
                    # as permanently retained history context. Subsequent rollout starts from start_chunk;
                    # the first frame always stays in the attention window (matching inference sink protection).
                    if sink_size > 0:
                        sink_latents = latents[:, :, :sink_size].to(weight_dtype)
                        sink_t_zero = torch.zeros(bsz, sink_size, device=latents.device, dtype=torch.long)
                        sink_seqlen = sink_size * frame_seqlen
                        with torch.no_grad():
                            transformer3d(
                                x=sink_latents, context=prompt_embeds, t=sink_t_zero,
                                seq_len=sink_seqlen,
                                mouse_actions=mouse_actions, keyboard_actions=keyboard_actions,
                                kv_caches=kv_caches,
                                current_start=0, frame_offset=0,
                            )

                    # c_local: local index in the generated chunk sequence (for exit_steps / grad decisions)
                    # f_start = rollout_start_frame + c_local * chunk_size
                    #   no sink: rollout_start_frame=0, equivalent to original c_idx * chunk_size
                    #   with sink: rollout_start_frame=1, first rollout chunk starts at frame 1
                    for c_local in range(generated_chunks):
                        f_start = rollout_start_frame + c_local * chunk_size
                        f_end = min(f_start + chunk_size, num_frames)
                        new_frames = f_end - f_start
                        chunk_noise = noise[:, :, f_start:f_end].clone()
                        exit_step = exit_steps[c_local]
                        is_grad_chunk = (c_local >= first_grad_chunk) and TRAIN_GENERATOR
                        current_start_token = f_start * frame_seqlen
                        seq_len_chunk = new_frames * frame_seqlen

                        # Start chunk from pure noise scaled by the first sigma
                        # Each chunk starts from pure noise; prior context comes only through the KV cache.
                        # chunk_noise here is the random initial noise for this chunk;
                        # GT latents are not used directly because DMD trains the generation distribution, not teacher-forcing MSE.
                        chunk_latents = chunk_noise.clone() * denoise_sigmas[0].to(chunk_noise.dtype)

                        # Snapshot cache indices: each denoising step writes the chunk's
                        # K/V at the same positions, so restore before every step.
                        # Multi-step denoising within the same chunk repeatedly forwards the current chunk.
                        # Each step temporarily writes KV cache; to prevent the previous step's temporary K/V
                        # from polluting the next step, restore the cache boundary before each denoise step.
                        snap_kv = _snapshot_caches(kv_caches)

                        x0_pred = None
                        for k, t_scalar in enumerate(denoise_timesteps):
                            t_cur = torch.full(
                                (bsz, new_frames), t_scalar.long().item(),
                                device=latents.device, dtype=torch.long,
                            )

                            _restore_caches(kv_caches, snap_kv)

                            is_exit = (k == exit_step)
                            # Only the exit step needs gradients for the generator.
                            # Non-exit steps only build the rollout trajectory to the exit point; no_grad saves memory.
                            with torch.set_grad_enabled(is_exit and is_grad_chunk):
                                v_pred_chunk = transformer3d(
                                    x=chunk_latents, context=prompt_embeds, t=t_cur,
                                    seq_len=seq_len_chunk,
                                    mouse_actions=mouse_actions, keyboard_actions=keyboard_actions,
                                    kv_caches=kv_caches,
                                    current_start=current_start_token, frame_offset=f_start,
                                )

                            # x0 prediction at current sigma: x0 = x_t - sigma * v (flow matching)
                            # flow matching predicts v = noise - data, so x0 = x_t - sigma * v.
                            if denoise_sigmas is not None:
                                sigma_at_step = denoise_sigmas[k].to(chunk_latents.dtype)
                            else:
                                sigma_at_step = (t_scalar.float() / 1000.0).to(chunk_latents.dtype)
                            x0_pred = chunk_latents - sigma_at_step * v_pred_chunk

                            if is_exit:
                                # Stop the chunk's few-step rollout at the random exit step.
                                # If this chunk is outside the grad window, detach it and use only as context for subsequent score steps.
                                chunk_outputs.append(x0_pred if is_grad_chunk else x0_pred.detach())
                                break

                            # Backward simulation (Self-Forcing): re-noise x0 with FRESH
                            # Instead of continuing with an ODE solver, re-noise x0 with fresh noise to the next sigma,
                            # matching Self-Forcing backward simulation.
                            # Each step constructs the next x_t from "current x0 estimate + fresh noise",
                            # reducing solver error accumulation and letting the student adapt to training-time randomness.
                            # noise at the next sigma instead of ODE step. This avoids ODE
                            # error accumulation and matches inference-time multistep distill.
                            with torch.no_grad():
                                if denoise_sigmas is not None:
                                    sigma_next = denoise_sigmas[k + 1].to(chunk_latents.dtype)
                                else:
                                    sigma_next = (denoise_timesteps[k + 1].float() / 1000.0).to(chunk_latents.dtype)
                                fresh_noise = torch.randn_like(x0_pred)
                                chunk_latents = ((1.0 - sigma_next) * x0_pred.detach()
                                                 + sigma_next * fresh_noise).to(weight_dtype)

                        # Clean pass at t=context_noise to update KV cache for next chunk
                        # After the current chunk produces x0, forward again with near-clean latents
                        # only to write KV cache; this cache becomes history for subsequent chunks.
                        # NOTE: this pass does not contribute to loss or update parameters; it simulates inference where
                        # "the current chunk is fixed as history context after denoising completes".
                        with torch.no_grad():
                            x0_context = x0_pred.detach()
                            if args.context_noise_level > 0:
                                ctx_sigma = args.context_noise_level
                                x0_context = (1.0 - ctx_sigma) * x0_context + ctx_sigma * torch.randn_like(x0_context)
                            ctx_t_zero = torch.zeros(bsz, new_frames, device=latents.device, dtype=torch.long)
                            _restore_caches(kv_caches, snap_kv)
                            transformer3d(
                                x=x0_context.to(weight_dtype), context=prompt_embeds, t=ctx_t_zero,
                                seq_len=seq_len_chunk,
                                mouse_actions=mouse_actions, keyboard_actions=keyboard_actions,
                                kv_caches=kv_caches,
                                current_start=current_start_token, frame_offset=f_start,
                            )

                # ===== Concat chunks for whole-video score evaluation =====
                # DMD score is not computed per chunk; generated chunks are concatenated into full latents
                # and evaluated by bidirectional real/fake score models for distribution mismatch.
                # This lets real/fake score assess global temporal quality, cross-chunk coherence, and text consistency;
                # the student's generation remains causal, but score supervision is whole-video level.
                # With sink training, GT sink frames are prepended: score models see the full video,
                # while sink frames appear as detached GT and do not affect student gradients.
                generated_video = torch.cat(chunk_outputs, dim=2)  # [B, C, F-sink, H, W]
                if sink_size > 0:
                    pred_image_full = torch.cat(
                        [latents[:, :, :sink_size].detach().to(weight_dtype), generated_video], dim=2
                    )  # [B, C, full_F, H, W]
                else:
                    pred_image_full = generated_video
                if debug_decode_now:
                    with torch.no_grad():
                        vae_unwrapped = accelerator.unwrap_model(vae)
                        pred_debug_latents = pred_image_full
                        pred_video = vae_unwrapped.decode(pred_debug_latents[:1].detach().to(vae_unwrapped.dtype)).sample
                        pred_video = ((pred_video / 2) + 0.5).clamp(0, 1).cpu()
                        debug_dir = os.path.join(args.output_dir, "debug_pred_decode")
                        os.makedirs(debug_dir, exist_ok=True)
                        debug_path = os.path.join(debug_dir, f"step-{global_step}-student_x0_laststep.mp4")
                        save_videos_grid(pred_video, debug_path, fps=16)
                total_F_full = pred_image_full.shape[2]
                seq_len_full = math.ceil(HW_patched * total_F_full)

                # ts_schedule: clamp DM/critic timestep range to generator's exit interval
                # score_min/max_step determines the re-noise strength range for pred_image_full.
                # If ts_schedule=True, the lower bound follows the range the student has already denoised to;
                # if ts_schedule_max=True, the upper bound also follows the student's entry timestep.
                # This concentrates score loss in the noise range the student is actually responsible for.
                if args.ts_schedule and denoised_timestep_to is not None:
                    score_min_step = max(min_score_step, denoised_timestep_to)
                else:
                    score_min_step = min_score_step
                if args.ts_schedule_max and denoised_timestep_from is not None:
                    score_max_step = min(max_score_step, max(denoised_timestep_from, score_min_step + 1))
                else:
                    score_max_step = max_score_step
                if score_min_step >= score_max_step:
                    score_min_step = min_score_step
                    score_max_step = max_score_step

                # ===== Whole-video DM loss (generator step only) =====
                # On generator update: re-noise student output, predict x0 with real/fake score respectively;
                # fake-real difference is the DMD approximate KL gradient.
                #
                # This section can be understood via the following math:
                #   1. student produces pred_image_full, a sample from the current generator distribution;
                #   2. randomly sample dm_t and re-noise pred_image_full to x_t;
                #   3. real_score gives x0_real: where the real distribution thinks x_t should denoise to;
                #   4. fake_score gives x0_fake_score: where the current student distribution thinks x_t should denoise to;
                #   5. grad = x0_fake_score - x0_real, the direction pushing student distribution toward real;
                #   6. construct MSE with target_dm = pred_image_full - grad so gradients flow back to student.
                #
                # NOTE: real_score/fake_score forwards are wrapped in no_grad:
                #   - generator step does not train score networks;
                #   - score networks only provide detached target directions;
                #   - gradients actually reach the student rollout behind pred_image_full.
                with accelerator.autocast():
                    if TRAIN_GENERATOR:
                        with torch.no_grad():
                            dm_t_raw = torch.randint(score_min_step, score_max_step, (bsz, 1), device=latents.device).long()
                            dm_t_raw = dm_t_raw.repeat(1, total_F_full)
                            dm_t_shifted = args.timestep_shift * (dm_t_raw.float() / 1000) / (1 + (args.timestep_shift - 1) * (dm_t_raw.float() / 1000)) * 1000
                            dm_t_shifted = dm_t_shifted.clamp(score_min_step, score_max_step).long()
                            # TI2V teacher conditioning: the sink frame(s) are the clean i2v
                            # anchor and MUST be fed clean (sigma=0) AND with timestep=0, exactly
                            # like Wan2.2-TI2V inference (per-frame timestep = mask * t). Feeding a
                            # noised sink frame at a high timestep is OOD for the teacher and
                            # produces degenerate x0_real.
                            if sink_size > 0:
                                dm_t_shifted[:, :sink_size] = 0
                            # Per-frame sigma [B, 1, F, 1, 1]: sink stays clean, the rest is noised.
                            dm_sigma_5d = (dm_t_shifted.float() / 1000.0)[:, None, :, None, None]
                            # The transformer consumes t as a PER-TOKEN tensor (length seq_len,
                            # f-major), not per-frame. Expand each frame's timestep over its
                            # frame_seqlen spatial tokens so the sink frame's tokens really get t=0.
                            dm_t_token = dm_t_shifted.repeat_interleave(frame_seqlen, dim=1)

                            # Re-noise the student sample to a random score timestep.
                            # detach is key here: constructing x_t via re-noising should not backprop through score models.
                            dm_noise = torch.randn_like(pred_image_full.detach())
                            x_t = (1.0 - dm_sigma_5d) * pred_image_full.detach() + dm_sigma_5d * dm_noise
                            x_t = x_t.to(weight_dtype)

                            # --- Real score with CFG ---
                            # real_score is the frozen teacher. Use conditional prompt and optional negative prompt
                            # to obtain CFG-augmented real-distribution velocity estimate.
                            v_real_cond = real_score_transformer3d(
                                x=x_t, context=prompt_embeds, t=dm_t_token, seq_len=seq_len_full,
                                **score_action_kwargs,
                            )
                            if args.real_guidance_scale > 0:
                                v_real_uncond = real_score_transformer3d(
                                    x=x_t, context=neg_prompt_embeds, t=dm_t_token, seq_len=seq_len_full,
                                    **score_action_kwargs,
                                )
                                # DMD CFG convention (matches official Self-Forcing model/dmd.py:108-110
                                # and configs/self_forcing_dmd.yaml:16 guidance_scale=3.0):
                                #   v = v_cond + w*(v_cond - v_uncond) = (1+w)*v_cond - w*v_uncond
                                # NOT the diffusers/sampling CFG convention:
                                #   v = v_uncond + w*(v_cond - v_uncond) = (1-w)*v_uncond + w*v_cond
                                # The two are isomorphic: DMD's w=N == sampling CFG w=N+1.
                                # So real_guidance_scale=3.0 here ≡ standard CFG=4.0 at sampling time.
                                v_real = v_real_cond + args.real_guidance_scale * (v_real_cond - v_real_uncond)
                            else:
                                v_real = v_real_cond
                            # Under flow matching, x0 = x_t - sigma * v.
                            # x0_real can be viewed as the clean video the teacher assigns to this noisy sample.
                            x0_real = x_t - dm_sigma_5d * v_real
                            if debug_decode_now:
                                vae_unwrapped = accelerator.unwrap_model(vae)
                                x_t_video = vae_unwrapped.decode(x_t[:1].detach().to(vae_unwrapped.dtype)).sample
                                x_t_video = ((x_t_video / 2) + 0.5).clamp(0, 1).cpu()
                                x0_real_video = vae_unwrapped.decode(x0_real[:1].detach().to(vae_unwrapped.dtype)).sample
                                x0_real_video = ((x0_real_video / 2) + 0.5).clamp(0, 1).cpu()
                                debug_dir = os.path.join(args.output_dir, "debug_pred_decode")
                                os.makedirs(debug_dir, exist_ok=True)
                                save_videos_grid(
                                    x_t_video,
                                    os.path.join(debug_dir, f"step-{global_step}-renoise_xt.mp4"),
                                    fps=16,
                                )
                                save_videos_grid(
                                    x0_real_video,
                                    os.path.join(debug_dir, f"step-{global_step}-real_teacher_x0.mp4"),
                                    fps=16,
                                )
                                # Fixed moderate-noise probe of the real-score target,
                                # independent of the random (shift-skewed) dm_t used for
                                # training. Diagnoses gray x0_real: if THIS is sharp the
                                # collapse is from extreme renoise (timestep_shift/max_score_step);
                                # if THIS is also gray the teacher itself can't score at this
                                # resolution/regime.
                                _dbg_t_val = 300
                                _dbg_t = torch.full((bsz, total_F_full), _dbg_t_val, device=latents.device, dtype=torch.long)
                                if sink_size > 0:
                                    _dbg_t[:, :sink_size] = 0
                                _dbg_sigma = (_dbg_t.float() / 1000.0)[:, None, :, None, None]
                                _dbg_t_token = _dbg_t.repeat_interleave(frame_seqlen, dim=1)
                                _dbg_noise = torch.randn_like(pred_image_full.detach())
                                _dbg_xt = ((1.0 - _dbg_sigma) * pred_image_full.detach() + _dbg_sigma * _dbg_noise).to(weight_dtype)
                                _dbg_v = real_score_transformer3d(
                                    x=_dbg_xt, context=prompt_embeds, t=_dbg_t_token, seq_len=seq_len_full,
                                    **score_action_kwargs,
                                )
                                if args.real_guidance_scale > 0:
                                    _dbg_v_unc = real_score_transformer3d(
                                        x=_dbg_xt, context=neg_prompt_embeds, t=_dbg_t_token, seq_len=seq_len_full,
                                        **score_action_kwargs,
                                    )
                                    _dbg_v = _dbg_v + args.real_guidance_scale * (_dbg_v - _dbg_v_unc)
                                _dbg_x0 = _dbg_xt - _dbg_sigma * _dbg_v
                                _dbg_video = vae_unwrapped.decode(_dbg_x0[:1].detach().to(vae_unwrapped.dtype)).sample
                                _dbg_video = ((_dbg_video / 2) + 0.5).clamp(0, 1).cpu()
                                save_videos_grid(
                                    _dbg_video,
                                    os.path.join(debug_dir, f"step-{global_step}-real_teacher_x0_t{_dbg_t_val}.mp4"),
                                    fps=16,
                                )

                            # --- Fake score ---
                            # fake_score is the current version of the trainable critic.
                            # In generator loss it is also not updated; it only estimates the "fake distribution score".
                            v_fake_cond = fake_score_transformer3d(
                                x=x_t, context=prompt_embeds, t=dm_t_token, seq_len=seq_len_full,
                                **score_action_kwargs,
                            )
                            if args.fake_guidance_scale > 0:
                                v_fake_uncond = fake_score_transformer3d(
                                    x=x_t, context=neg_prompt_embeds, t=dm_t_token, seq_len=seq_len_full,
                                    **score_action_kwargs,
                                )
                                # Same DMD CFG convention as real_score above (Self-Forcing model/dmd.py:87-89).
                                # Default fake_guidance_scale=0.0 means no CFG on fake_score (matches official).
                                v_fake = v_fake_cond + args.fake_guidance_scale * (v_fake_cond - v_fake_uncond)
                            else:
                                v_fake = v_fake_cond
                            # x0_fake_score is where the critic thinks the student/fake distribution will denoise to.
                            x0_fake_score = x_t - dm_sigma_5d * v_fake

                            # --- KL gradient with normalization (DMD eq. 7-8) ---
                            # grad points to the difference between fake and real distributions; normalizer prevents gradient scale explosion.
                            # If fake_score and real_score agree, grad is near 0, meaning student distribution
                            # is close to teacher; if the gap is large, generator is pushed toward real_score.
                            grad = x0_fake_score - x0_real
                            p_real = pred_image_full.detach() - x0_real
                            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
                            grad = grad / normalizer
                            grad = torch.nan_to_num(grad)

                        # Convert the hand-crafted DMD gradient into ordinary MSE via a stop-gradient target.
                        # With target_dm detached, backprop is equivalent to moving pred_image_full along -grad.
                        target_dm = (pred_image_full - grad).detach()
                        # Exclude the sink frame: it is the clean i2v anchor (detached GT, sigma=0)
                        # and carries no DMD gradient for the student.
                        if sink_size > 0:
                            dm_loss = 0.5 * F.mse_loss(
                                pred_image_full[:, :, sink_size:].float(),
                                target_dm[:, :, sink_size:].float(),
                            )
                        else:
                            dm_loss = 0.5 * F.mse_loss(pred_image_full.float(), target_dm.float())
                        total_gen_loss = args.dm_loss_weight * dm_loss

                    # ===== Whole-video critic loss =====
                    # On critic update: treat student output as fake data, re-noise and apply standard flow matching,
                    # so fake score learns the denoising direction of the student's current distribution.
                    # This step is fake_score's supervision source:
                    #   - pred_image_full.detach() is treated as "fake clean data";
                    #   - re-noise to get x_t_fs;
                    #   - flow matching target velocity is noise - data;
                    #   - fake_score_transformer3d predicts v_pred_fs and is fit with MSE.
                    # Because pred_image_full is detached, critic updates do not backprop into the student.
                    with torch.no_grad():
                        pred_image_detached = pred_image_full.detach()
                        fs_t_raw = torch.randint(score_min_step, score_max_step, (bsz, 1), device=latents.device).long()
                        fs_t_raw = fs_t_raw.repeat(1, total_F_full)
                        fs_t_shifted = args.timestep_shift * (fs_t_raw.float() / 1000) / (1 + (args.timestep_shift - 1) * (fs_t_raw.float() / 1000)) * 1000
                        fs_t_shifted = fs_t_shifted.clamp(score_min_step, score_max_step).long()
                        # TI2V: critic must see the same conditioning as the teacher — keep the
                        # sink frame clean (sigma=0) and timestep=0.
                        if sink_size > 0:
                            fs_t_shifted[:, :sink_size] = 0
                        fs_sigma_5d = (fs_t_shifted.float() / 1000.0)[:, None, :, None, None]
                        # Per-token timestep (see generator DM loss): expand over frame_seqlen.
                        fs_t_token = fs_t_shifted.repeat_interleave(frame_seqlen, dim=1)

                        fs_noise = torch.randn_like(pred_image_detached)
                        x_t_fs = (1.0 - fs_sigma_5d) * pred_image_detached + fs_sigma_5d * fs_noise
                        v_target_fs = fs_noise - pred_image_detached
                        x_t_fs = x_t_fs.to(weight_dtype)

                    # No CFG here: the critic must first learn the score of the student distribution itself.
                    # fake_guidance_scale is optionally used only in fake_score evaluation for generator DM loss.
                    v_pred_fs = fake_score_transformer3d(
                        x=x_t_fs, context=prompt_embeds, t=fs_t_token, seq_len=seq_len_full,
                        **score_action_kwargs,
                    )
                    # Exclude the sink frame: it is the clean i2v anchor (sigma=0, t=0) and has
                    # no meaningful flow-matching velocity target (would be a degenerate target).
                    if sink_size > 0:
                        total_critic_loss = F.mse_loss(
                            v_pred_fs[:, :, sink_size:].float(),
                            v_target_fs[:, :, sink_size:].float(),
                        )
                    else:
                        total_critic_loss = F.mse_loss(v_pred_fs.float(), v_target_fs.float())
                    total_critic_loss_val = total_critic_loss.item()

                # Final loss aggregation (whole-video, no /num_chunks divide)
                # student loss is whole-video DM loss, not averaged per chunk;
                # pred_image_full already contains all chunks from the current rollout.
                if TRAIN_GENERATOR:
                    loss = total_gen_loss
                else:
                    # On critic-only steps the student should not update, but accelerator/deepspeed still needs
                    # a zero loss tied to student parameters to maintain the distributed training flow.
                    loss = sum(0.0 * p.sum() for p in network.parameters() if p.requires_grad)

                step_gen_loss = total_gen_loss.item() if TRAIN_GENERATOR else 0.0
                step_critic_loss = total_critic_loss_val

                loss = loss.mean()
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_gen_loss += step_gen_loss / args.gradient_accumulation_steps
                train_critic_loss += step_critic_loss / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Critic backward + step (after DeepSpeed backward to avoid distributed deadlock)
                # fake_score_network is not passed through accelerator.prepare; manually all_reduce gradients in distributed mode.
                # Complete student/DeepSpeed backward+step first, then update critic separately,
                # to avoid communication deadlock from backward ordering of two optimizers/wrappers.
                if step % args.gradient_accumulation_steps == 0:
                    fake_score_optimizer.zero_grad()
                critic_loss_avg = total_critic_loss / float(args.gradient_accumulation_steps)
                critic_loss_avg.backward()
                if accelerator.sync_gradients:
                    if torch.distributed.is_initialized():
                        for p in fake_score_network.parameters():
                            if p.grad is not None:
                                torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG)
                    torch.nn.utils.clip_grad_norm_(fake_score_network.parameters(), args.max_grad_norm)
                    fake_score_optimizer.step()

                # EMA update (only on generator steps)
                # Update EMA only on steps where the student generator actually updates.
                if TRAIN_GENERATOR:
                    if ema_shadow is None and global_step >= args.ema_start_step:
                        ema_shadow = {name: param.data.clone() for name, param in network.named_parameters() if param.requires_grad}
                        print(f"[EMA] Started tracking at step {global_step}")
                    if ema_shadow is not None:
                        update_ema(ema_shadow, network, args.ema_decay)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({
                    "train_gen_loss": train_gen_loss,
                    "train_critic_loss": train_critic_loss,
                    "lr": lr_scheduler.get_last_lr()[0],
                }, step=global_step)
                step_type = "G+C" if TRAIN_GENERATOR else "C"
                progress_bar.set_postfix(
                    gen=f"{train_gen_loss:.4f}",
                    crt=f"{train_critic_loss:.4f}",
                    type=step_type,
                )
                train_gen_loss = 0.0
                train_critic_loss = 0.0

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
                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        accelerator_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        save_training_checkpoint(accelerator_save_path)
                        logger.info(f"Saved checkpoint to {accelerator_save_path}")

                if accelerator.is_main_process:
                    if args.validation_steps > 0 and global_step % args.validation_steps == 0:
                        ema_backup = None
                        if ema_shadow is not None:
                            ema_backup = apply_ema(ema_shadow, network)

                        one_sample = {
                            "pixel_values": batch["pixel_values"][0:1],
                            "text": batch["text"][0:1],
                            "idx": batch["idx"][0:1],
                            "mouse_actions": batch["mouse_actions"][0:1],
                            "keyboard_actions": batch["keyboard_actions"][0:1],
                        }

                        log_validation(
                            one_sample,
                            vae, text_encoder, tokenizer,
                            transformer3d, network,
                            config, args,
                            accelerator, weight_dtype, global_step,
                            mouse_scale=args.mouse_scale,
                        )

                        if ema_backup is not None:
                            restore_from_ema(ema_backup, network)

            logs = {"gen_loss": step_gen_loss, "crt_loss": step_critic_loss, "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        save_training_checkpoint(accelerator_save_path)
        logger.info(f"Saved final checkpoint to {accelerator_save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()