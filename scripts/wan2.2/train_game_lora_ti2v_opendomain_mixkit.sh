# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
export MODEL_NAME="ckpts/models/Diffusion_Transformer/Wan2.2-TI2V-5B"
# Open-domain (Mixkit) bidirectional teacher: Wan2.2 base + trainable LoRA.
# Trained on datasets/mixkit at 352x640 so the DMD real-score teacher's motion prior
# matches the resolution and the Mixkit distribution used for distillation.
export DATASET_NAME="datasets/mixkit"
export DATASET_META_NAME="datasets/mixkit/video_captions_mixkit.csv"

# Validation = the DMD eval case (image + text), i2v conditioning on the first frame.
export VAL_IMAGE="asset/street_night.jpg"
export VAL_PROMPT="First-person perspective walking down a lively city street at night. Neon signs and bright billboards glow on both sides, cars drive past with headlights and taillights streaking slightly."

# --- single-node defaults (override for multi-node) ---
export GPUS_PER_NODE="${GPUS_PER_NODE:-7}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export RANK="${RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
# Skip the flaky GPU node (GPU[2]); gfx950 stability fixes.
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,3,4,5,6,7}"
export TORCH_BLAS_PREFER_HIPBLASLT=0
export XFORMERS_IGNORE_FLASH_VERSION_CHECK=1

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7

accelerate launch --num_processes $((GPUS_PER_NODE*WORLD_SIZE)) --num_machines $WORLD_SIZE --machine_rank=$RANK --mixed_precision="bf16" \
  --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT \
  --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard \
  scripts/wan2.2/train_game_lora_ti2v.py \
  --config_path="config/wan2.2/wan_civitai_5b.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=1024 \
  --video_sample_size 352 640 \
  --token_sample_size=512 \
  --video_sample_stride=1 \
  --video_sample_n_frames=81 \
  --train_batch_size=4 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="output_dir_opendomain_teacher_mixkit" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --use_deepspeed \
  --random_hw_adapt \
  --training_with_video_token_length \
  --random_frame_crop \
  --uniform_sampling \
  --report_to "tensorboard" \
  --low_vram \
  --tracker_project_name "OpenDomain-Teacher-Mixkit-wan2.2" \
  --checkpoints_total_limit=5 \
  --validation_steps=100 \
  --validation_image_start "$VAL_IMAGE" \
  --validation_prompt "$VAL_PROMPT" \
  --validation_num_inference_steps=30 \
  --validation_guidance_scale=5.0 \
  --validation_shift=5.0 \
  --validation_fps=16
