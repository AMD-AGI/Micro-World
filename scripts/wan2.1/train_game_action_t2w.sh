# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B"
export DATASET_NAME="datasets"
export DATASET_META_NAME="datasets/dataset/video_captions_v0.csv"

accelerate launch --num_processes $((GPUS_PER_NODE*WORLD_SIZE)) --num_machines $WORLD_SIZE --machine_rank=$RANK --mixed_precision="bf16" \
  --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT \
  --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard \
  scripts/wan2.1/train_game_action_t2w.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=1024 \
  --video_sample_size 352 640 \
  --token_sample_size=512 \
  --video_sample_stride=1 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=1e-05 \
  --lr_scheduler="cosine_with_restarts" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=1e-1 \
  --adam_epsilon=1e-8 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --use_deepspeed \
  --random_hw_adapt \
  --training_with_video_token_length \
  --random_frame_crop \
  --uniform_sampling \
  --network_alpha 12 \
  --low_vram \
  --tracker_project_name "MicroWorld-Game" \
  --checkpoints_total_limit=5 \
  --trainable_modules "action" \
  --lora_skip_name "action" \
  --validation_steps=100 \
  --motion_sub_loss \
  --lora_path "models/T2W/lora_diffusion_pytorch_model.safetensors" \
  --train_mode "adaln" #"controlnet"


