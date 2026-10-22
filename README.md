<div align="center">
  <br>
  <br>
  <h1>AMD Micro-World: action-controlled Interactive world model</h1>
<a href='https://rocm.blogs.amd.com/artificial-intelligence/Micro-World-V2/README.html'><img src='https://img.shields.io/badge/2.0%20Technical-Blog-red'></a> 
<a href='https://rocm.blogs.amd.com/artificial-intelligence/micro-world/README.html'><img src='https://img.shields.io/badge/1.0%20Technical-Blog-red'></a> 
<a href='https://huggingface.co/amd/Micro-World-2'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-2.0 Model-blue'></a>
<a href='https://huggingface.co/amd/Micro-World-T2W'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-1.0%20T2W Model-blue'></a>
<a href='https://huggingface.co/amd/Micro-World-I2W'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-1.0%20I2W Model-blue'></a>
<a href='https://huggingface.co/datasets/amd/Micro-World-Capsaicin'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-2.0%20Capsaicin%20Dataset-blue'></a>
<a href='https://huggingface.co/datasets/amd/Micro-World-MC-Dataset'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-1.0%20Minecraft%20Dataset-blue'></a>
</div>

# Table of Contents
- [Introduction](#introduction)
- [Micro-World 2.0](#micro-world-20)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Micro-World 2.0 Pipeline](#micro-world-20-pipeline)
- [Inference](#inference)
- [Interactive demo](#interactive-demo)
- [Inference optimizations (2.0)](#inference-optimizations-20)
- [Micro-World 1.0](#micro-world-10)
- [Evaluation](#evaluation)
- [Video Result](#video-result)
- [Acknowledgement](#acknowledgement)

# Introduction
Micro-World is a series of action-controlled interactive world models developed by the AMD AIG team and trained on AMD Instinct™ MI250/MI325 GPUs. It includes both text-to-video and image-to-video variants, enabling a wide range of application scenarios. Built on Wan as the base model, Micro-World is trained on a MineCraft dataset and is designed to generate high-quality, open-domain visual environments.

# Micro-World 2.0

2.0 is under active development. It adds a wan2.2-based causal generation stack
on top of the released 1.0 model, without changing 1.0 behaviour: every 1.0
entry point under `examples/wan2.1` and `scripts/wan2.1` still runs as
documented below, and the new behaviour is reached through explicit flags.
Weights are at [amd/Micro-World-2](https://huggingface.co/amd/Micro-World-2).

## What 2.0 adds

- **Causal chunked generation.** `CausalWanTransformer3DModel` replaces
  whole-clip denoising with block-causal attention over latent frame chunks and
  a rolling KV cache, and `AutoencoderKLWan3_8.cached_decode` decodes one latent
  frame at a time. Together these let a rollout stream frames instead of
  materialising the whole clip, so clip length is no longer bounded by memory.
- **Few-step distillation.** DMD training and the matching
  `WanCausalDMDPipeline` bring generation down to a handful of steps per chunk.
- **Sequence-parallel inference.** `microworld.parallel` wraps xfuser's
  Ulysses/Ring parallelism; calling `enable_multi_gpus_inference()` shards the
  sequence across GPUs. Without it, and without xfuser installed, models run
  exactly as in 1.0.
- **Open-domain data.** Loaders and preparation tooling for Mixkit and Capsaicin,
  beyond 1.0's Minecraft data.
- **Action-aware distillation.** The DMD teacher and critic are themselves
  action-conditioned, which is what keeps the few-step student responsive to input.

## Layout

| Path | Contents |
| --- | --- |
| `microworld/` | The library. `models/wan_causal_*` and `pipeline/pipeline_wan_causal*` are 2.0; the rest is shared with 1.0. |
| `microworld/parallel/` | Sequence-parallel and FSDP helpers. |
| `examples/wan2.1`, `scripts/wan2.1` | 1.0 inference and training, unchanged. |
| `scripts/wan2.2/` | The five 2.0 training stages and the final inference script. One `.py` trainer per stage, one `.sh` launcher per run. |
| `examples/wan2.2/` | 2.0 inference entry point, plus `demo.py` for the interactive rollout. |
| `distillation_data/` | Mixkit download and transcode. |
| `tools/` | Data metadata fixes and sample visualisation. See `tools/README.md`. |
| `test_metrics/` | Image quality, action controllability and temporal stability metrics. |

## Compatibility with 1.0

Shared code gained 2.0 behaviour only behind opt-in switches, so 1.0 defaults are
unchanged:

| Switch | Default (1.0) | 2.0 use |
| --- | --- | --- |
| `WanActionAdaLNModel(max_position_index=...)` | `None`, no clamp | causal models pass `84`, their training horizon, to keep long rollouts in distribution |
| `ActionModule.forward(frame_offset=...)` | `0` | the global frame index of a chunk during rollout |
| `VideoGameDataset(frame_scaling=...)` | `"crop"` | `"resize"` keeps the full field of view for open-domain clips |
| `VideoGameDataset(enforce_sample_n_frames=...)` | `False`, uses the annotated window | `True` forces a uniform clip length |
| `parse_config(cursor_base_scale=..., clamp_cursor_to_frame=...)` | `0.4`, `False` | `1.0`, `True` for a legible cursor over long rollouts |

`enable_multi_gpus_inference()` is likewise opt-in: `sp_world_size` stays 1 until
it is called, and every parallel code path is behind `sp_world_size > 1`.

# Installation
Our model is build on AMD GPU and ROCm enviroment.

## a. From docker
We strongly recommend docker enviroment.

```
# build image
docker build -t microworld:latest .

# enter image
docker run -it --rm --name=agent --network=host \
  --device=/dev/kfd --device=/dev/dri --group-add=video --group-add=render \
  --ipc=host \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_SOCKET_IFNAME=lo \
  -e NCCL_SOCKET_FAMILY=AF_INET \
  -e NCCL_DEBUG=WARN \
  -e RCCL_MSCCLPP_ENABLE=0 \
  -e RCCL_MSCCL_ENABLE=0 \
  -e NCCL_MIN_NCHANNELS=16 -e NCCL_MAX_NCHANNELS=32 \
  microworld:latest
```

## b. Conda
```
conda create -n AMD_microworld python=3.12
conda activate AMD_microworld

# Install torch/torchvision
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/rocm6.4

# Important: Install ROCm flash-attn by offical guideline https://github.com/Dao-AILab/flash-attention
pip install -r requirements.txt
```

# Data Preparation

2.0 trains on two datasets, one open-domain and one action-labelled. Every stage
reads them through `VideoGameDataset`, which resolves a clip as
`<train_data_dir>/[<dir>/]video/<original video name>`, so the `video` subdirectory
name is fixed and is usually a symlink.

## Base weights

Download the 2.0 checkpoints from [amd/Micro-World-2](https://huggingface.co/amd/Micro-World-2).
The Hub paths are the directory names below; `--local-dir ckpts` places them
where the scripts expect them. The Wan2.2 base is not in that repo.

```bash
hf download amd/Micro-World-2 --local-dir ckpts
hf download Wan-AI/Wan2.2-TI2V-5B --local-dir ckpts/models/Diffusion_Transformer/Wan2.2-TI2V-5B
```

`causal-forcing-init` passed as `--transformer_path` in stages 1, 2 and 5.

| Hub path | Local path | What it is |
| --- | --- | --- |
| — | `ckpts/models/Diffusion_Transformer/Wan2.2-TI2V-5B` | Wan2.2-TI2V-5B base, from Wan-AI |
| `causal-forcing-init/model.pt` | `ckpts/causal-forcing-init/model.pt` | Our causal-forcing initialization |
| `causal-lora/lora_diffusion_pytorch_model.safetensors` | `ckpts/causal-lora/lora_diffusion_pytorch_model.safetensors` | Stage 1 base LoRA |
| `action/` | `ckpts/action/` | Stage 2 causal action module |
| `dmd-lora/lora_diffusion_pytorch_model.safetensors` | `ckpts/dmd-lora/lora_diffusion_pytorch_model.safetensors` | Stage 5 DMD student LoRA |
| `light_vae/` | `ckpts/light_vae/` | Optional TurboVAED decoder |

## Mixkit (open domain, no action labels)

Supplies first frames, prompts and the open-domain motion prior. Used by stages 1,
3 and 5.

```
# 1. download and extract -> datasets/mixkit/videos/<Category>/
python distillation_data/download_mixkit.py --local_dir datasets/mixkit

# 2. transcode every category to 640x352 @16fps -> datasets/mixkit/videos_resized
bash distillation_data/run_mixkit.sh

# 3. build the caption csv and the datasets/mixkit/video -> videos_resized symlink
python -m distillation_data.build_mixkit_meta

# 4. keep only static, camera-drivable categories (used by DMD distillation)
python tools/filter_mixkit_meta.py \
    --src datasets/mixkit/video_captions_mixkit.csv \
    --dst datasets/mixkit/video_captions_mixkit_filtered.csv
```

## Capsaicin (first-person capture with action labels)

Supplies the per-frame keyboard/mouse labels that the action module learns from, and
in stage 5 the pool of trajectories sampled at random. Expected layout, with
`video` a symlink to the shipped `videos` folder:

```
datasets/capsaicin/200f/
├── video -> videos/          # clips
├── metadata-detection/       # per-clip action jsons
└── video_captions_v0.csv     # prompts
```

Repair the reversed W/S labels once before training, then check what the dataloader
actually produces:

```
python tools/fix_ws_swap.py --dir datasets/capsaicin/200f/metadata-detection
python tools/viz_train_sample.py          # capsaicin sample + action overlay
python tools/viz_mixkit_dmd_sample.py     # mixkit frames + random capsaicin actions
```

# Micro-World 2.0 Pipeline

The final model is a 4-step causal action-controlled student, distilled from an
action-aware bidirectional teacher. Getting there takes five training stages: two
build the causal student, two build the teacher, and the last distils one into the
other.

| # | Stage | Launcher | Trainer | Trains | Produces |
| --- | --- | --- | --- | --- | --- |
| 1 | Causal LoRA on Mixkit | `train_causal_lora_ti2v_cvid.sh` | `train_causal_lora_ti2v.py` | LoRA (rank 256) | base LoRA |
| 2 | Action module on Capsaicin | `train_causal_ti2v_action_capsaicin.sh` | `train_causal_ti2v_action.py` | action adaLN only | causal action module |
| 3 | Bidirectional teacher on Mixkit | `train_game_lora_ti2v_opendomain_mixkit.sh` | `train_game_lora_ti2v.py` | LoRA | teacher LoRA |
| 4 | Teacher action module | `train_teacher_action_capsaicin.sh` | `train_teacher_ti2v_action.py` | action adaLN only | teacher action module |
| 5 | DMD distillation | `train_causal_dmd_ti2v_action_teacher_mixkit_161f.sh` | `train_causal_dmd_ti2v_action_teacher.py` | student LoRA + critic | 4-step student LoRA |
| 6 | Inference | `predict_causal_action_dmd_mix.sh` | `examples/wan2.2/predict_causal_action_evalcase.py` | — | rollout videos |

Everything runs at 352x640. Stages 1-4 train on 81-frame clips, stage 5 on 161
frames (`161 = 4*40+1` satisfies the Wan2.2-VAE 4x temporal compression), and
stage 6 rolls out to 1001 frames.

The launchers carry the checkpoint paths as `export`ed variables at the top. Each
stage writes to a timestamped subdirectory, so after finishing a stage you have to
point the next stage's variable at the checkpoint you picked.

## Stage 1: causal LoRA on Mixkit

```
bash scripts/wan2.2/train_causal_lora_ti2v_cvid.sh
```

Load order is base -> causal init (`--transformer_path`) -> trainable LoRA. This
adapts the causal backbone to 352x640 open-domain video before any action
conditioning exists. Result: `output_dir_capsaicin_causal_lora_cvid/<run>/checkpoint-N/lora_diffusion_pytorch_model.safetensors`.
The released copy is `ckpts/causal-lora/lora_diffusion_pytorch_model.safetensors`,
referred to below as the **base LoRA**. It is consumed by stages 2, 5 and 6.

## Stage 2: action module on Capsaicin

```
bash scripts/wan2.2/train_causal_ti2v_action_capsaicin.sh
```

Load order is base -> causal init -> base LoRA -> action init, and
`--trainable_modules="action"` freezes everything except `action_preprocess` and the
per-block `action_adaLN_modulation`. Because the backbone is frozen, the module
learns action control without disturbing the open-domain prior from stage 1.

Chunk-wise attention is configured here and must stay consistent through stages 5
and 6: `--chunk_size=3 --sink_size=1 --local_attn_size=21`. Result:
`output_dir_capsaicin_action_v1/<run>/checkpoint-N`. The released copy is
`ckpts/action/`, the **causal action module**.

## Stage 3: bidirectional teacher on Mixkit

```
bash scripts/wan2.2/train_game_lora_ti2v_opendomain_mixkit.sh
```

The DMD real score has to be a strong bidirectional model at the target resolution,
so this trains a LoRA on the vanilla (non-causal) Wan2.2 base over the same Mixkit
data. Result: `output_dir_opendomain_teacher_mixkit/...`, the **teacher LoRA**, used
by stages 4 and 5 as `TEACHER_LORA_PATH`.

## Stage 4: teacher action module

```
bash scripts/wan2.2/train_teacher_action_capsaicin.sh
```

The teacher LoRA is merged and frozen, and only the action adaLN modules are trained
on Capsaicin, warm-started from stage 2 so mainly the per-block modulation is
recalibrated to the teacher. This is what makes the teacher action-aware: an
action-blind teacher gives the student no gradient for following actions, and the
distilled student then stops responding to input. Result:
`output_dir_capsaicin_teacher_action_81/<run>/checkpoint-N`, the **teacher action
module**.

## Stage 5: DMD distillation

```
bash scripts/wan2.2/train_causal_dmd_ti2v_action_teacher_mixkit_161f.sh
```

Both sides are assembled from the earlier stages:

| Role | Composition | Flags |
| --- | --- | --- |
| Student (trained) | base -> causal init -> base LoRA -> action module -> trainable student LoRA | `--transformer_path`, `--lora_path`, `--action_pretrain_path` |
| Real score + critic | base -> teacher LoRA -> teacher action module | `--teacher_lora_path`, `--teacher_action_pretrain_path` |

Data is decoupled: frames and prompts come from the filtered Mixkit csv, while the
action trajectory is sampled at random from Capsaicin's json pool
(`--action_pool_root`), since Mixkit has no action labels. The student learns the
4-step schedule `--denoising_step_list="1000,750,500,250"`, and the critic is
updated `--dfake_gen_update_ratio=5` times per generator step. Result:
`output_dir_mixkit_dmd_action_teacher_161f_v1/...`. The released copy is
`ckpts/dmd-lora/lora_diffusion_pytorch_model.safetensors`, the **student LoRA**.

# Inference

```
bash scripts/wan2.2/predict_causal_action_dmd_mix.sh
```

Four steps per chunk, no CFG (`--guidance_scale 1.0`), rolled out to 1001 frames.
The load order mirrors training: base -> causal init -> base LoRA -> action module ->
student LoRA. Point `BASE_LORA_PATH`, `ACTION_CHECKPOINT_PATH` and
`STUDENT_LORA_PATH` at your stage 1, 2 and 5 checkpoints; clearing
`ACTION_CHECKPOINT_PATH` runs plain TI2V with no action control.

Useful overrides:

| Variable | Meaning |
| --- | --- |
| `ACTION_PRESETS` | Which trajectories to render, one video each. `all` runs every preset; `w`/`s`/`a`/`d` and `mouse_*` are the single-action probes. |
| `VALIDATION_IMAGE_START`, `PROMPT` | The starting frame and its caption. `cliff1001` is authored for `asset/cliff.jpg`. |
| `VIDEO_LENGTH` | Frames to roll out. Long clips stay in memory because the VAE decodes in chunks of `VAE_DECODE_CHUNK_FRAMES` latent frames while carrying its cache. |
| `MOUSE_MAG`, `MOUSE_SCALE` | Per-frame mouse magnitude. Values far above the training range make the model extrapolate into unstable rotations. |
| `REAL_ACTION_NPZ` | Replay a recorded trajectory instead of a preset. Produce it with `tools/extract_real_case.py`. |

For a live rollout in the browser instead of a fixed preset, use the
[interactive demo](#interactive-demo).

## Interactive demo

`examples/wan2.2/demo.py` serves a Flask-SocketIO page that streams a causal
action rollout chunk by chunk. The load order is the same as
`scripts/wan2.2/predict_causal_action_dmd_mix.sh`: Wan2.2-TI2V-5B,
`causal-forcing-init` checkpoint, causal LoRA, action adaLN, then the DMD student LoRA. Weight loading
takes a couple of minutes; the port accepts connections after that.

Run it from the repository root, inside the ROCm container. The environment
variables are the tier-F generator optimizations (see below) and are read at
import time, so they have to be set on the same command.

```bash
cd /workspace/Sprite

TRANSFORMER_PATH=ckpts/causal-forcing-init/model.pt
BASE_LORA_PATH=ckpts/causal-lora/lora_diffusion_pytorch_model.safetensors
ACTION_CHECKPOINT_PATH=ckpts/action
STUDENT_LORA_PATH=ckpts/dmd-lora/lora_diffusion_pytorch_model.safetensors

CUDA_VISIBLE_DEVICES=0 \
WAN_ADALN_FUSION=script WAN_RMSNORM_FUSION=script WAN_QKV_FUSION=1 \
WAN_CTX_CACHE=1 WAN_ATTN_AITER=1 GPU_ARCHS=gfx950 \
python -u examples/wan2.2/demo.py \
  --model_name ckpts/models/Diffusion_Transformer/Wan2.2-TI2V-5B \
  --causal_model_pt "$TRANSFORMER_PATH" \
  --student_base_lora_path "$BASE_LORA_PATH" \
  --action_checkpoint_path "$ACTION_CHECKPOINT_PATH" \
  --student_lora_path "$STUDENT_LORA_PATH" \
  --light_vae ckpts/light_vae \
  --video_length 401 --fps 24 --seed 42 \
  --height 352 --width 640 \
  --num_inference_steps 4 --guidance_scale 1.0 \
  --save_dir samples/demo \
  --host 0.0.0.0 --port 5002
```

Omit `--light_vae` to decode with the Wan VAE. `--host 0.0.0.0` makes the page
reachable at `http://<machine>:5002`.

### Page

Open that URL in a normal browser tab and keep the tab in the foreground.
Embedded or background views clamp the playback timer to about one frame per
second, which shows up as a realtime FPS of 1 even while the GPU is ahead.

| Field | Meaning |
| --- | --- |
| Prompt | Caption for the rollout. |
| Seed / CFG / Steps | Defaults `42` / `1.0` / `4`. Keep CFG at `1.0` so the DMD student runs without classifier-free guidance. |
| FPS | Playback rate of the viewer. It does not change how fast chunks are denoised. Default `24`. |
| Video Length | Pixel frames to generate. Default `401`. |
| First Frame Path | Image path relative to the repo root, for image-to-video. Leave it empty for text-to-video. Example: `asset/sana/game_style_001.png`. |
| Action List | JSON trajectory. **Start** begins a run, **Stop** requests a stop, **Replay** plays the frames already received. |

The action list is a JSON array. Each entry except the last is
`[end_frame, "w s a d shift ctrl unused mouse_y mouse_x"]`. The segment runs
from the previous end frame (frame 1 for the first segment) through `end_frame`.
`w`/`s`/`a`/`d`/`shift`/`ctrl` are 0 or 1. `mouse_y` and `mouse_x` are degrees
per generated frame; negative `mouse_y` looks up, negative `mouse_x` looks left.
The last element is a string of frame indices where the space bar is held, or
`""` when space is unused:

```json
[
  [80, "1 0 0 0 0 0 0 0 0"],
  ""
]
```

While frames arrive, the keyboard panel and the mouse cursor show the action of
the frame on screen. `realtime FPS` counts frames actually drawn in the last
second, so it drops while the queue is empty.

The first rollout on a given GPU compiles MIOpen kernels for the lightweight
decoder. That stall is once per process; later **Start** clicks reuse the
compiled kernels. Editing `examples/wan2.2/templates/demo.html` requires a
process restart, because Flask caches templates when debug mode is off.

## Inference optimizations (2.0)

2.0 causal inference ships a ladder of optional speedups for the diffusion
generator and VAE decode. Every switch is **opt-in** and read from environment
variables at **import time**, so you must export them **before** starting the
Python process (or prefix them on the command line). With all switches unset,
behaviour is identical to the pre-optimization code path.

These optimizations target **streaming inference** (small per-chunk tensors).
Leave them unset during training.

### Optimization tiers

Each tier adds one item on top of the previous one:

| Tier | What it adds | Switch | Equivalence |
| --- | --- | --- | --- |
| A | Baseline (all off) | — | — |
| B | AdaLN elementwise fusion (`modulate` + `gated_residual`) | `WAN_ADALN_FUSION=script` | bit-exact |
| C | RMSNorm fusion | `WAN_RMSNORM_FUSION=script` | bit-exact |
| D | Self-attn QKV / cross-attn KV projection GEMM fusion | `WAN_QKV_FUSION=1` | bit-exact |
| E | Prompt context cache (text embedding + cross-attn k/v across steps/chunks) | `WAN_CTX_CACHE=1` | bit-exact |
| F | AITER varlen FMHA attention backend | `WAN_ATTN_AITER=1` | numerically equivalent |
| G | Lightweight TurboVAED decoder (decode only) | `--light_vae <weights_dir>` | lossy |

Tiers A–E are bit-exact versus the original path. F swaps the flash-attn
Composable Kernel backend for AITER's FMHA: not bit-identical, but same
information (deviation vs SDPA is smaller than CK). G replaces only the VAE
decoder and trades a small amount of decode fidelity for ~4× faster decode.

Additional always-on code paths (no env var): non-overlapping `patchify_via_linear`,
on-device sequence lengths, and `embed_context()` for cross-attention.

### Recommended production config (tier F)

```bash
export WAN_ADALN_FUSION=script
export WAN_RMSNORM_FUSION=script
export WAN_QKV_FUSION=1
export WAN_CTX_CACHE=1
export WAN_ATTN_AITER=1   # requires amd-aiter; falls back to CK if missing

bash scripts/wan2.2/predict_causal_action_dmd_mix.sh
```

DMD few-step inference should use `--guidance_scale 1.0` (no CFG), as in the
launcher above.

Install AITER on ROCm when enabling tier F:

```bash
pip install amd-aiter
```

Optional self-check flags (cross-check fused paths against eager on every call):
`WAN_ADALN_FUSION_SELFCHECK=1`, `WAN_RMSNORM_FUSION_SELFCHECK=1`,
`WAN_QKV_FUSION_SELFCHECK=1`, `WAN_CTX_CACHE_SELFCHECK=1`.

### Lightweight VAE decoder (tier G, optional)

Tier G swaps only the **VAE decode** step: the Wan encoder and latent space are
unchanged (48 channels, 4× temporal / 16× spatial compression). A smaller
**TurboVAED** decoder from
[amd/HummingbirdXT](https://huggingface.co/amd/HummingbirdXT/tree/main/vae/wan22_v1_tiling_16_12)
replaces the original Wan2.2 decoder for roughly **4× faster decode**, at the
cost of some decode fidelity (lossy — verify output before deploying).

| Component | Location |
| --- | --- |
| Model definition | `microworld/models/turbo_vaed_wan22.py` (`AutoencoderKLTurboVAED`) |
| Streaming wrapper | `microworld/models/light_vae_decoder.py` (`LightVaeDecoder`) |
| Inference flag | `--light_vae <weights_dir>` on `predict_causal_action_evalcase.py` |

**Weights directory.** `--light_vae` points at a folder that contains exactly:

- `config.json`
- `diffusion_pytorch_model.safetensors`

`amd/Micro-World-2` already contains `light_vae/`. To fetch that decoder on its
own, download the HummingbirdXT release into `ckpts/light_vae/`:

```bash
hf download amd/HummingbirdXT \
  vae/wan22_v1_tiling_16_12/config.json \
  vae/wan22_v1_tiling_16_12/diffusion_pytorch_model.safetensors \
  --local-dir ckpts/light_vae
```

**Streaming behaviour.** TurboVAED is non-causal and has no Wan VAE `feat_cache`.
`LightVaeDecoder` adapts it for chunked rollout decode: each chunk prepends the
previous chunk's last latent frame as left context so per-chunk pixel output
aligns with the original causal VAE. Only decoder weights are loaded; the
encoder path is untouched.

**Usage** (typically on top of tier F):

```bash
export WAN_ADALN_FUSION=script WAN_RMSNORM_FUSION=script WAN_QKV_FUSION=1 \
       WAN_CTX_CACHE=1 WAN_ATTN_AITER=1

python examples/wan2.2/predict_causal_action_evalcase.py \
  ... \
  --light_vae ckpts/light_vae
```

Omit `--light_vae` (or pass an empty string) to keep the original Wan VAE
decoder. TurboVAED depends on `timm`, already listed in `requirements.txt`.

### Benchmark per tier

Because switches are process-global, measure each tier in a **separate**
process:

```bash
# Tier A baseline (all off)
python3 examples/wan2.2/bench_optimizations.py --tag A

# Tier F (all generator optimizations on)
WAN_ADALN_FUSION=script WAN_RMSNORM_FUSION=script WAN_QKV_FUSION=1 \
WAN_CTX_CACHE=1 WAN_ATTN_AITER=1 \
python3 examples/wan2.2/bench_optimizations.py --tag F
```

Results are written under `samples/bench/`. Use `--dump_latents` to save
latents for cross-tier equivalence checks.

Implementation details and env-var semantics live in
`microworld/models/fused_ops.py`.

# Micro-World 1.0

The released 1.0 text-to-world and image-to-world models are unchanged and still run
from their own entry points under `examples/wan2.1` and `scripts/wan2.1`.

## Inference
- **Step 1**: Download the corresponding model weights and place them in the `models` folder.
```
# Download T2W model weights:
hf download amd/Micro-World-T2W --local-dir models/T2W

# Download I2W model weights:
hf download amd/Micro-World-I2W --local-dir models/I2W
```
- **Step 2**: We provide action-controlled model inference scripts under the `examples` folder.
  - Modify config in the script, such as `transformer_path`, `lora_path`, `prompt`, `neg_prompt`, `GPU_memory_mode`, `validation_image_start`, `action_list`.
    - `validation_image_start` is the reference image path of image-to-video generation.
    - `action_list` follows formation of [[{end frame}, "w, s, a, d, shift, ctrl, _, mouse_y, mouse_x"],...,"{space frames}"]. For example, [[10, "0 1 0 0 0 0 0 0 0"], [80, "0 0 0 1 0 0 0 0 0"], "30 65"] indicates press s from frame 0 to frame 10, press d from frame 10 to frame 80, and press space at frame 30 and 65.
  - For example, you can run T2W action controled model inference using following command:
```
python examples/wan2.1/predict_t2w_action_control.py
```

## Training
We have provided our collected minecraft action controlled dataset, the game lora and action controlled model weights. 
- **Step 1**: Download your target *base model* weights and place them in the `models` folder.
```
# Download Wan2.1 T2V model weights:
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir models/Diffusion_Transformer/Wan2.1-T2V-1.3B

# Download pretrained model weights and lora
hf download amd/Micro-World-T2W --local-dir models/T2W

# Download Wan2.1 I2V model weights:
hf download Wan-AI/Wan2.1-I2V-14B-480P --local-dir models/Diffusion_Transformer/Wan2.1-I2V-14B-480P

# Download pretrained model weights and lora
hf download amd/Micro-World-I2W --local-dir models/I2W

# Download train dataset:
hf download amd/Micro-World-MC-Dataset --local-dir datasets --repo-type=dataset
```
- **Step 2**: We provide action-controlled model training scripts under the `scripts` folder.
  - Modify config in the bash script.
  - For example, you can run T2W action controled model training using following command:
```
bash scripts/wan2.1/train_game_action_t2w.sh
```

# Evaluation
We follow the evaluation protocol from [Matrix-Game](https://github.com/SkyworkAI/Matrix-Game) to assess image quality, action controllability, and temporal stability.

- **Step 1** Download the IDM model and weights from [VPT](https://github.com/openai/Video-Pre-Training), and place them under `test_metrics/idm_model`.
- **Step 2** Clone [GameWorldScore](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-1/GameWorldScore) and move the folder into `test_metrics/GameWorldScore`.
- **Step 3** Run the evaluation script.

```
cd test_metrics
python evaluate.py \
        --model idm_moddel/4x_idm.model \
        --weights idm_moddel/4x_idm.weights \
        --visual_metrics \ # evaluate image quality(PSNR, LPIPS, FVD)
        --control_metrics \ # evaluate action Controllability 
        --temporal_metrics \ # evaluate temporal stability
        --infer-demo-num 0 --n-frames 81 \
        --video-path your_testing_output_path \
        --ref-path dataset_val/video \
        --output-file "metrics_log/ground_truth/idm_res_pred.json" \
        --json-path dataset_val/metadata-detection/
```

# Video Result
## T2W Model
### In Domain
<table border="0" style="width: 100%; table-layout: fixed; text-align: center; margin-top: 20px;">
  <tr>
      <td style="vertical-align: top; width: 33%;">
          <video src="https://github.com/user-attachments/assets/01ecff57-5fc8-40c0-b7c1-1c72525b598c" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; overflow:hidden; font-size: 14px;">
        W
        </div>
      </td>
       <td style="vertical-align: top; width: 33%;">
          <video src="https://github.com/user-attachments/assets/0156af1f-5fe2-4276-9cec-ba97b2476018" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; font-size: 14px;">
        S
        </div>
     </td>
      <td style="vertical-align: top; width: 33%;">
          <video src="https://github.com/user-attachments/assets/d27268e5-9fbc-49f7-b3ca-882fb58f21b6" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; font-size: 14px;">
        A
        </div>
      </td>
  </tr>
</table>

<table border="0" style="width: 100%; table-layout: fixed; text-align: center; margin-top: 20px;">
  <tr>
     <td style="vertical-align: top; width: 33%;">
          <video src="https://github.com/user-attachments/assets/aff52ef1-0c9c-4a03-961f-6aa5361b636d" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; font-size: 14px;">
        D
        </div>
     </td>
      <td style="vertical-align: top; width: 33%;">
          <video src="https://github.com/user-attachments/assets/b5d37d89-5cf0-40a5-8504-61f68e944fb9" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; font-size: 14px;">
        W+Ctrl
        </div>
      </td>
      <td style="vertical-align: top; width: 33%;">
          <video src="https://github.com/user-attachments/assets/1b0d50c8-a037-4671-a146-b77672260322" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; font-size: 14px;">
        W+Shift
        </div>
     </td>
  </tr>
</table>

<table border="0" style="width: 100%; table-layout: fixed; text-align: center; margin-top: 20px;">
  <tr>
    <td style="vertical-align: top; width: 33%;">
          <video src="https://github.com/user-attachments/assets/b13a14d7-5882-42dd-872b-8f61b9ab7060" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; font-size: 14px;">
        Multiple control
        </div>
     </td>
     <td style="vertical-align: top; width: 33%;">
          <video src="https://github.com/user-attachments/assets/1218bbda-7993-4075-881b-2e16002acda8" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; font-size: 14px;">
        Mouse down and up
        </div>
     </td>
      <td style="vertical-align: top; width: 33%;">
          <video src="https://github.com/user-attachments/assets/31471313-d94b-4936-b23b-12e7f89fda87" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; font-size: 14px;">
        Mouse right and left
        </div>
      </td>
  </tr>
</table>

### Open Domain
<table border="0" style="width: 100%; table-layout: fixed; text-align: center; margin-top: 20px;">
  <tr>
      <td style="vertical-align: top; width: 33%;">
          <video src="https://github.com/user-attachments/assets/25ec4ba8-4f65-4b26-8966-13437647f240" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; text-align: left;">
            <details>
              <summary style="cursor: pointer; font-size: 13px;">View Prompt</summary>
              <div style="font-size: 12px; margin-top: 5px; color: #555;">
                A cozy living room with sunlight streaming through window, vintage furniture, soft shadows.
              </div>
            </details>
          </div>        
      </td>
      <td style="vertical-align: top; width: 33%;">
          <video src="https://github.com/user-attachments/assets/a92149a8-6c4d-4b9a-8ada-81b47b4c81e7" width="100%" controls autoplay loop></video>
        <div style="margin-top: 8px; text-align: left;">
            <details>
              <summary style="cursor: pointer; font-size: 13px;">View Prompt</summary>
              <div style="font-size: 12px; margin-top: 5px; color: #555;">
                A cozy living room with sunlight streaming through window, vintage furniture, soft shadows.
              </div>
            </details>
          </div>  
      </td>
       <td style="vertical-align: top; width: 33%;">
          <video src="https://github.com/user-attachments/assets/67b35842-04fd-4a0f-9a5c-6914d9f77e66" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; text-align: left;">
            <details>
              <summary style="cursor: pointer; font-size: 13px;">View Prompt</summary>
              <div style="font-size: 12px; margin-top: 5px; color: #555;">
                Running along a cliffside path in a tropical island in first person perspective, with turquoise waters crashing against the rocks far below, the salty scent of the ocean carried by the breeze, and the sound of distant waves blending with the calls of seagulls as the path twists and turns along the jagged cliffs.
              </div>
            </details>
          </div>  
     </td>
  </tr>
</table>

<table border="0" style="width: 100%; table-layout: fixed; text-align: center; margin-top: 20px;">
  <tr>
      <td style="vertical-align: top; width: 33%;">
          <video src="https://github.com/user-attachments/assets/d4a46b8b-022d-4fca-964f-c1d477111f4e" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; text-align: left;">
            <details>
              <summary style="cursor: pointer; font-size: 13px;">View Prompt</summary>
              <div style="font-size: 12px; margin-top: 5px; color: #555;">
                A young bear stands next to a large tree in a grassy meadow, its dark fur catching the soft daylight. The bear seems poised, observing its surroundings in a tranquil landscape, with rolling hills and sparse trees dotting the background under a pale blue sky.
              </div>
            </details>
          </div>  
     </td>
     <td style="vertical-align: top; width: 33%;">
          <video src="https://github.com/user-attachments/assets/b6f77a1a-58ce-43db-b6c5-efe09b7a9142" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; text-align: left;">
            <details>
              <summary style="cursor: pointer; font-size: 13px;">View Prompt</summary>
              <div style="font-size: 12px; margin-top: 5px; color: #555;">
                A giant panda rests peacefully under a blooming cherry blossom tree, its black and white fur contrasting beautifully with the delicate pink petals. The ground is lightly sprinkled with fallen blossoms, and the tranquil setting is framed by the soft hues of the blossoms and the grassy field surrounding the tree.
              </div>
            </details>
          </div>  
     </td>      
     <td style="vertical-align: top; width: 33%;">
          <video src="https://github.com/user-attachments/assets/c9225344-8b0b-4249-ab77-c8e5c4dddacc" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; text-align: left;">
            <details>
              <summary style="cursor: pointer; font-size: 13px;">View Prompt</summary>
              <div style="font-size: 12px; margin-top: 5px; color: #555;">
                Exploring an ancient jungle ruin in first person perspective surrounded by towering stone statues covered in moss and vines.
              </div>
            </details>
          </div> 
     </td>      
  </tr>
</table>

## I2W Model
We observe that fully decoupling the action module from game-specific styles in large-scale models remains challenging. As a result, we apply both the LoRA weights and the action module during inference for the I2W results.
<table border="0" style="width: 100%; table-layout: fixed; text-align: center; margin-top: 20px;">
  <tr>
      <td style="vertical-align: top; width: 50%;">
          <video src="https://github.com/user-attachments/assets/f135d5c9-0379-4ace-bf22-1671cef261af" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; text-align: left;">
            <details>
              <summary style="cursor: pointer; font-size: 13px;">View Prompt</summary>
              <div style="font-size: 12px; margin-top: 5px; color: #555;">
                First-person perspective walking down a lively city street at night. Neon signs and bright billboards glow on both sides, cars drive past with headlights and taillights streaking slightly. camera motion directly aligned with user actions, immersive urban night scene.
              </div>
            </details>
          </div>        
      </td>
      <td style="vertical-align: top; width: 50%;">
          <video src="https://github.com/user-attachments/assets/2088d2da-95a6-4908-b7a2-f60458281b5e" width="100%" controls autoplay loop></video>
        <div style="margin-top: 8px; text-align: left;">
            <details>
              <summary style="cursor: pointer; font-size: 13px;">View Prompt</summary>
              <div style="font-size: 12px; margin-top: 5px; color: #555;">
                First-person perspective standing in front of an ornate traditional Chinese temple. The symmetrical facade features red lanterns, intricate carvings, and a curved tiled roof decorated with dragons. Bright daytime lighting, consistent environment, camera motion directly aligned with user actions, immersive and interactive exploration.
              </div>
            </details>
          </div>  
      </td>
  </tr>
</table>

<table border="0" style="width: 100%; table-layout: fixed; text-align: center; margin-top: 20px;">
  <tr>
       <td style="vertical-align: top; width: 50%;">
          <video src="https://github.com/user-attachments/assets/9e1185cc-5480-4059-8643-7b6e08fff0c1" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; text-align: left;">
            <details>
              <summary style="cursor: pointer; font-size: 13px;">View Prompt</summary>
              <div style="font-size: 12px; margin-top: 5px; color: #555;">
                First-person perspective of standing in a rocky desert valley, looking at a camel a few meters ahead. The camel stands calmly on uneven stones, its long legs and single hump clearly visible. Bright midday sunlight, dry air, muted earth tones, distant barren mountains. Natural handheld camera feeling, camera motion controlled by user actions, smooth movement, cinematic realism.
              </div>
            </details>
          </div>  
     </td>
     <td style="vertical-align: top; width: 50%;">
          <video src="https://github.com/user-attachments/assets/c75d7344-7016-494e-be00-103d28e43738" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; text-align: left;">
            <details>
              <summary style="cursor: pointer; font-size: 13px;">View Prompt</summary>
              <div style="font-size: 12px; margin-top: 5px; color: #555;">
                First-person perspective walking through a narrow urban alley, old red brick industrial buildings on both sides, cobblestone street stretching forward with strong depth, metal walkways connecting buildings above, overcast daylight, soft diffused lighting, cool and muted color tones, quiet and empty environment, no people, camera motion controlled by user actions, smooth movement, stable horizon, realistic scale and geometry, high realism, cinematic urban scene.
              </div>
            </details>
          </div>  
     </td>
  </tr>
</table>

<table border="0" style="width: 100%; table-layout: fixed; text-align: center; margin-top: 20px;">
  <tr>
     <td style="vertical-align: top; width: 50%;">
          <video src="https://github.com/user-attachments/assets/f6da97af-0d3a-4b6a-b80f-5ae3c03ccbf6" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; text-align: left;">
            <details>
              <summary style="cursor: pointer; font-size: 13px;">View Prompt</summary>
              <div style="font-size: 12px; margin-top: 5px; color: #555;">
                First-person perspective coastal exploration scene, walking along a cliffside stone path with wooden railings, green bushes lining the walkway, ocean to the left with gentle waves, distant islands visible under a clear sky, realistic head-mounted camera view, smooth forward motion, stable horizon, natural human eye level, high realism, consistent environment, camera motion directly aligned with user actions, immersive and interactive exploration.
              </div>
            </details>
          </div>  
     </td>      
     <td style="vertical-align: top; width: 50%;">
          <video src="https://github.com/user-attachments/assets/b76a8aca-d1da-47ba-88e9-3da36f64429d" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; text-align: left;">
            <details>
              <summary style="cursor: pointer; font-size: 13px;">View Prompt</summary>
              <div style="font-size: 12px; margin-top: 5px; color: #555;">
                First-person perspective inside a cozy living room, walking around a warm fireplace, soft carpet underfoot, furniture arranged neatly, bookshelves, plants, and warm table lamps on both sides, warm indoor lighting, calm and quiet atmosphere, natural head-level camera movement, camera motion driven by user actions, realistic scale and depth, high realism, cinematic lighting, no people, no distortion.
              </div>
            </details>
          </div> 
     </td>      
  </tr>
</table>

# 🤗Resources
### Technical blog
- [Micro-World: First AMD Open-Source World Models for Interactive Video Generation](https://rocm.blogs.amd.com/artificial-intelligence/micro-world/README.html)

### Pre-trained models
- Micro-World 2.0: [amd/Micro-World-2](https://huggingface.co/amd/Micro-World-2) (`causal-forcing-init`, `causal-lora`, `action`, `dmd-lora`, `light_vae`)
- Text-to-World (1.0): [Micro-World-T2W](https://huggingface.co/amd/Micro-World-T2W)
- Image-to-World (1.0): [Micro-World-I2W](https://huggingface.co/amd/Micro-World-I2W)
### Dataset
- [MC-Dataset](https://huggingface.co/datasets/amd/Micro-World-MC-Dataset)

# Acknowledgement
Our codebase is built upon [Wan2.1](https://github.com/Wan-Video/Wan2.1/), [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun). We sincerely thank the authors for open-sourcing their excellent codebases. 

Our datasets are collected using [MineDojo](https://github.com/MineDojo/MineDojo) and captioned with [miniCPM-V](https://github.com/OpenBMB/MiniCPM-V). We also extend our appreciation to the respective teams for their high-quality tools and contributions.