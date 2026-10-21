# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
export MODEL_NAME="ckpts/models/Diffusion_Transformer/Wan2.2-TI2V-5B"

# Capsaicin 200f data + captions. DATASET_NAME must point at the dataset folder so the
# dataset builds paths as <root>/video/<id> and <root>/metadata-detection/<json>.
# (The repo dataset hardcodes the "video" subdir; capsaicin ships a "videos" folder,
# so datasets/capsaicin/200f/video is a symlink -> videos.)
export DATASET_NAME="datasets/capsaicin/200f"
export DATASET_META_NAME="datasets/capsaicin/200f/video_captions_v0.csv"

# Base load order (mirrors inference): base -> CAUSAL init -> base LoRA -> action (adaLN only).
export CAUSAL_MODEL_PT="ckpts/causal-forcing-init/model.pt"
# Action module init (only its adaLN modules are loaded): the open-domain-adapted causal action
# (same checkpoint DMD uses), so we fine-tune the latest action on DL3DV rather than the old init.
export ACTION_PRETRAIN_PATH="ckpts/external/yugeng12/codes/Sprite/output_dir/20260514-0503-unmerge_lora/checkpoint-14450/"
export BASE_LORA_PATH="ckpts/causal-lora/lora_diffusion_pytorch_model.safetensors"
export VALIDATION_IMAGE_START="asset/street_night.jpg"
export VALIDATION_PROMPT="First-person perspective walking down a lively city street at night. Neon signs and bright billboards glow on both sides, cars drive past with headlights and taillights streaking slightly."
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
  scripts/wan2.2/train_causal_ti2v_action.py \
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
  --output_dir="output_dir_capsaicin_action_v1" \
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
  --chunk_size=3 \
  --sink_size=1 \
  --local_attn_size=21 \
  --transformer_path=$CAUSAL_MODEL_PT \
  --action_pretrain_path=$ACTION_PRETRAIN_PATH \
  --base_lora_path=$BASE_LORA_PATH \
  --validation_image_start=$VALIDATION_IMAGE_START \
  --validation_prompt="$VALIDATION_PROMPT" \
  --validation_num_inference_steps=30 \
  --validation_guidance_scale=6.0 \
  --validation_shift=5.0 \
  --validation_fps=16 \
  --validation_steps=100 \
  --trainable_modules="action"
