# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
#!/usr/bin/env bash
set -euo pipefail

# 4-step DMD inference, no CFG (guidance_scale=1.0).
# Same base load order as the eval case: base -> CAUSAL init -> base LoRA -> action (adaLN).
# Set ACTION_CHECKPOINT_PATH="" to run plain TI2V without action control.

export XFORMERS_IGNORE_FLASH_VERSION_CHECK=1
MODEL_NAME="${MODEL_NAME:-ckpts/models/Diffusion_Transformer/Wan2.2-TI2V-5B}"
TRANSFORMER_PATH="${TRANSFORMER_PATH:-ckpts/causal-forcing-init/model.pt}"
ACTION_CHECKPOINT_PATH="${ACTION_CHECKPOINT_PATH:-ckpts/action}"

BASE_LORA_PATH="${BASE_LORA_PATH:-ckpts/causal-lora/lora_diffusion_pytorch_model.safetensors}"
# Optional DMD student LoRA (the 4-step distilled weights). Leave empty to skip.
STUDENT_LORA_PATH="${STUDENT_LORA_PATH-ckpts/dmd-lora/lora_diffusion_pytorch_model.safetensors}"
# Active case: asset/cliff.jpg -- a seaside clifftop walkway. Pairs with the
# cliff1001 action preset, which is authored for this image.
VALIDATION_IMAGE_START="${VALIDATION_IMAGE_START:-asset/cliff.jpg}"
PROMPT="${PROMPT:-First-person view walking along a stone-paved clifftop path by the sea. A reddish-brown wooden fence and trimmed green hedges line the left edge, beyond which steep cliffs drop to the ocean where waves roll gently toward the shore. Trees, a green lamppost and a red bench sit along the right side, with a hazy bright sky and distant headlands on the horizon. Warm, calm daylight, smooth steady camera motion.}"

OUTPUT_DIR="${OUTPUT_DIR:-samples/dmd-161f_ckpt2000}"
# Action presets to test, one video each. 'all' renders every preset; the single-action
# probes are w / s / a / d / mouse_x_pos / mouse_x_neg / mouse_y_pos / mouse_y_neg.
ACTION_PRESETS="${ACTION_PRESETS:-cliff1001}"

# Number of latent frames decoded per VAE call (1 latent frame = 4 video frames).
# The VAE cache is carried across chunks, so the output matches a single full decode
# while GPU peak memory during decoding stays flat in the clip length. Lower it further
# if the decode still OOMs; 0 restores the old single-call decode.
VAE_DECODE_CHUNK_FRAMES="${VAE_DECODE_CHUNK_FRAMES:-8}"

# Test-time mouse magnitude (per-frame degrees fed to the action model).
MOUSE_MAG="${MOUSE_MAG:-0.8}"
MOUSE_SCALE="${MOUSE_SCALE:-1.0}"
# UI cursor visualization scale. cursor_vis_scale = cursor_vis_ref / (mouse_mag*mouse_scale),
# so setting cursor_vis_ref = mouse_mag*mouse_scale gives cursor_vis_scale=1.0, i.e. the
# UI cursor is drawn at the SAME real-delta scale as tools/viz_train_sample.py (cursor_gain=3
# in the pipeline). The previous default (cursor_vis_ref=0.05) squashed the cursor ~12x.
CURSOR_VIS_REF="${CURSOR_VIS_REF:-$(awk "BEGIN{print $MOUSE_MAG*$MOUSE_SCALE}")}"

if [[ ! -f "$VALIDATION_IMAGE_START" ]]; then
  echo "Validation image not found: $VALIDATION_IMAGE_START"
  exit 1
fi

python examples/wan2.2/predict_causal_action_evalcase.py \
  --model_name "$MODEL_NAME" \
  --action_checkpoint_path "$ACTION_CHECKPOINT_PATH" \
  --transformer_path "$TRANSFORMER_PATH" \
  --base_lora_path "$BASE_LORA_PATH" \
  --student_lora_path "$STUDENT_LORA_PATH" \
  --validation_image_start "$VALIDATION_IMAGE_START" \
  --prompt "$PROMPT" \
  --video_height "${VIDEO_HEIGHT:-352}" \
  --video_width "${VIDEO_WIDTH:-640}" \
  --video_length "${VIDEO_LENGTH:-1001}" \
  --vae_decode_chunk_frames "$VAE_DECODE_CHUNK_FRAMES" \
  --use_dmd_pipeline \
  --num_inference_steps "${NUM_INFERENCE_STEPS:-4}" \
  --denoising_step_list ${DENOISING_STEP_LIST:-1000 750 500 250} \
  --warp_denoising_step "${WARP_DENOISING_STEP:-1}" \
  --num_train_timestep "${NUM_TRAIN_TIMESTEP:-1000}" \
  --context_noise_level "${CONTEXT_NOISE_LEVEL:-0.0}" \
  --guidance_scale "${GUIDANCE_SCALE:-1.0}" \
  --shift "${SHIFT:-5.0}" \
  --chunk_size "${CHUNK_SIZE:-3}" \
  --sink_size "${SINK_SIZE:-1}" \
  --local_attn_size "${LOCAL_ATTN_SIZE:-21}" \
  --action_presets ${ACTION_PRESETS} \
  --mouse_mag "$MOUSE_MAG" \
  --mouse_scale "$MOUSE_SCALE" \
  --cursor_vis_ref "$CURSOR_VIS_REF" \
  --real_action_npz "${REAL_ACTION_NPZ:-}" \
  --seed "${SEED:-42}" \
  --fps "${FPS:-16}" \
  --dtype "${DTYPE:-bf16}" \
  --sampler_name "${SAMPLER_NAME:-Flow_Unipc}" \
  --gpu_memory_mode "${GPU_MEMORY_MODE:-model_full_load}" \
  --output_dir "$OUTPUT_DIR"
