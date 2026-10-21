# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
export MODEL_NAME="ckpts/models/Diffusion_Transformer/Wan2.2-TI2V-5B"

# Capsaicin 200f data + captions + per-clip action labels (same data as the causal action run).
export DATASET_NAME="datasets/capsaicin/200f"
export DATASET_META_NAME="datasets/capsaicin/200f/video_captions_v0.csv"

# Teacher = vanilla bidirectional Wan2.2 base + teacher LoRA (frozen, merged). Only the action
# adaLN modules are trained, warm-started from the existing 81f action checkpoint so the action
# encoder is reused and mainly the per-block modulation is recalibrated to the teacher.
export TEACHER_LORA_PATH="ckpts/capsaicin_teacher_1300/lora_diffusion_pytorch_model.safetensors"
export ACTION_PRETRAIN_PATH="ckpts/output_dir_capsaicin_action_200f_81"

export VALIDATION_IMAGE_START="asset/cliff.jpg"
export VALIDATION_PROMPT="First-person view walking forward along a stone-paved clifftop path by the sea. A red wooden fence and green hedges line the left edge above the ocean, gentle waves rolling toward the shore far below, calm daylight, smooth steady camera motion."

export XFORMERS_IGNORE_FLASH_VERSION_CHECK=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,3,4,5,6,7}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-7}"
export WORLD_SIZE=${WORLD_SIZE:-1}
export RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}

accelerate launch --num_processes $((GPUS_PER_NODE*WORLD_SIZE)) --num_machines $WORLD_SIZE --machine_rank=$RANK --mixed_precision="bf16" \
  --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT \
  --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard \
  scripts/wan2.2/train_teacher_ti2v_action.py \
  --config_path="config/wan2.2/wan_civitai_5b.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size 352 640 \
  --video_sample_stride=1 \
  --video_sample_n_frames=81 \
  --enable_random_temporal_sample \
  --temporal_reverse_ratio=0.5 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=1000 \
  --checkpointing_steps=50 \
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="output_dir_capsaicin_teacher_action_81" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=1.0 \
  --uniform_sampling \
  --text_drop_ratio=0.0 \
  --drop_camera_sentences \
  --checkpoints_total_limit=5 \
  --i2v_ratio=1.0 \
  --teacher_lora_path=$TEACHER_LORA_PATH \
  --action_pretrain_path=$ACTION_PRETRAIN_PATH \
  --validation_image_start=$VALIDATION_IMAGE_START \
  --validation_prompt="$VALIDATION_PROMPT" \
  --validation_num_inference_steps=30 \
  --validation_guidance_scale=5.0 \
  --validation_shift=5.0 \
  --validation_fps=16 \
  --validation_steps=100 \
  --trainable_modules="action"
