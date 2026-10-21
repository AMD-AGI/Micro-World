# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
export MODEL_NAME="ckpts/models/Diffusion_Transformer/Wan2.2-TI2V-5B"
# Mixkit open-domain data. Resized clips live in datasets/mixkit/videos_resized;
# a `video` symlink (datasets/mixkit/video -> videos_resized) is created by
# distillation_data/build_mixkit_meta.py, since the loader hardcodes the `video`
# subdir. Each meta row's `original video name` is "<Category>/<file>.mp4".
export DATASET_NAME="datasets/mixkit"
export DATASET_META_NAME="datasets/mixkit/video_captions_mixkit.csv"
export CAUSAL_MODEL_PT="ckpts/causal-forcing-init/model.pt"
# Two validation cases: a real night-street scene and a Minecraft grassland scene.
export VAL_IMAGE_1="asset/street_night.jpg"
export VAL_PROMPT_1="First-person perspective walking down a lively city street at night. Neon signs and bright billboards glow on both sides, cars drive past with headlights and taillights streaking slightly."
export VAL_IMAGE_2="asset/first_frame.png"
export VAL_PROMPT_2="The video shows a Minecraft grassland biome during the day. Tall green grass blocks stretch across rolling hills, with blocky trees and a clear blue sky. First-person view moving forward through the plains."
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7
# We rebuilt flash-attn to 2.8.4 for gfx950; xformers only "supports" <=2.8.3 via
# a hard version check. This flag skips that check (2.8.4 API is compatible).
export XFORMERS_IGNORE_FLASH_VERSION_CHECK=1
# Random-step "GPU Hang" on gfx950 (MI350) with fixed input shapes: hipBLASLt GEMM
# kernels are a common culprit on new archs. Fall back to rocBLAS for stability.
export TORCH_BLAS_PREFER_HIPBLASLT=0

# Skip the flaky GPU that repeatedly triggered GPU Hang: KFD "node-4" maps to
# PCI 0000:5B:00.0 = HIP GPU[2] (verified via KFD location_id 0x5B00). Train on 7 GPUs.
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,3,4,5,6,7}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-7}"
export WORLD_SIZE=${WORLD_SIZE:-1}
export RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}

accelerate launch --num_processes $((GPUS_PER_NODE*WORLD_SIZE)) --num_machines $WORLD_SIZE --machine_rank=$RANK --mixed_precision="bf16" \
  --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT \
  --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard \
  scripts/wan2.2/train_causal_lora_ti2v.py \
  --config_path="config/wan2.2/wan_civitai_5b.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --transformer_path=$CAUSAL_MODEL_PT \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --rank=256 \
  --network_alpha=128 \
  --video_sample_size 352 640 \
  --video_sample_stride=1 \
  --video_sample_n_frames=81 \
  --train_batch_size=8 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-04 \
  --seed=42 \
  --output_dir="output_dir_capsaicin_causal_lora_cvid" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.5 \
  --uniform_sampling \
  --text_drop_ratio=0.0 \
  --drop_camera_sentences \
  --checkpoints_total_limit=5 \
  --i2v_ratio=1.0 \
  --chunk_size=3 \
  --validation_image_start "$VAL_IMAGE_1" "$VAL_IMAGE_2" \
  --validation_prompt "$VAL_PROMPT_1" "$VAL_PROMPT_2" \
  --validation_num_inference_steps=30 \
  --validation_guidance_scale=6.0 \
  --validation_shift=5.0 \
  --validation_fps=16 \
  --validation_steps=50
