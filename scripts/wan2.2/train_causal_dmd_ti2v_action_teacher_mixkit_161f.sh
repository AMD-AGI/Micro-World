# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
export XFORMERS_IGNORE_FLASH_VERSION_CHECK=1
export MODEL_NAME="ckpts/models/Diffusion_Transformer/Wan2.2-TI2V-5B"

# DMD distillation of the causal action student, trained on 161-frame clips.
# 161 = 4*40+1 satisfies the Wan2.2-VAE 4x temporal-compression constraint.
#
# Decoupled data: Mixkit supplies the first frame + prompt (diverse open-domain content),
# while the action trajectory is drawn AT RANDOM from the Capsaicin 200f action pool.
# Mixkit has no per-clip actions, so we point --action_pool_root at Capsaicin's
# metadata-detection json folder; the dataset then samples a random capsaicin action
# window per __getitem__ while frames/prompt still come from --train_data_meta (Mixkit).
export DATASET_NAME="datasets/mixkit"
# Filtered meta: only static, camera-drivable scene categories are kept
# (Street/City/beach/sea/sky/forest/mountain/night/water/snow/clouds/sunset/sun/earth/rain).
# Subject-dominated / dynamic clips (People/Sport/Dance/animals/effects...) are dropped,
# since the motion here is a first-person camera trajectory that only fits static scenes.
# Regenerate with tools/filter_mixkit_meta.py after changing the whitelist.
export DATASET_META_NAME="datasets/mixkit/video_captions_mixkit_filtered.csv"
export ACTION_POOL_ROOT="datasets/capsaicin/200f/metadata-detection"

# Student base load order (mirrors inference / action training):
#   base -> CAUSAL init (model.pt) -> base LoRA -> action (adaLN only) -> trainable student LoRA
export CAUSAL_MODEL_PT="ckpts/causal-forcing-init/model.pt"
export ACTION_CHECKPOINT_PATH="ckpts/action"
export BASE_LORA_PATH="ckpts/causal-lora/lora_diffusion_pytorch_model.safetensors"

# Teacher / real-score = ACTION-AWARE bidirectional teacher:
#   vanilla Wan2.2 bidirectional base + teacher LoRA + action (adaLN) modules,
#   i.e. the teacher LoRA from train_game_lora_ti2v_opendomain_mixkit.sh plus the
#   action modules from train_teacher_action_capsaicin.sh.
# Both the real score AND the fake-score critic get this base + action, so the DMD gradient
# rewards the student for following actions (an action-blind teacher can't, which is why the
# previous distilled student stopped following actions).
export TEACHER_LORA_PATH="ckpts/capsaicin_teacher_1300/lora_diffusion_pytorch_model.safetensors"
export TEACHER_ACTION_CHECKPOINT_PATH="output_dir_capsaicin_teacher_action_81/20260728-0659/checkpoint-1200"

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7
# export COMET_API_KEY="xx7XRDFBYe9aUUjhOdfFS8fnk"

# Train on GPU ids 1-7 (7 cards). ROCm uses HIP_VISIBLE_DEVICES; accelerate then
# sees these as local devices 0-6, so num_processes must match the card count.
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,3,4,5,6,7}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-7}"
export WORLD_SIZE=${WORLD_SIZE:-1}
export RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}

accelerate launch --num_processes $((GPUS_PER_NODE*WORLD_SIZE)) --num_machines $WORLD_SIZE --machine_rank=$RANK --mixed_precision="bf16" \
  --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT \
  --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard \
  scripts/wan2.2/train_causal_dmd_ti2v_action_teacher.py \
  --config_path="config/wan2.2/wan_civitai_5b.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --action_pool_root=$ACTION_POOL_ROOT \
  --video_sample_size 352 640 \
  --video_sample_stride=1 \
  --video_sample_n_frames=161 \
  --enable_random_temporal_sample \
  --temporal_reverse_ratio=0.0 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="output_dir_mixkit_dmd_action_teacher_161f_v1" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=1.0 \
  --checkpoints_total_limit=5 \
  --validation_steps=50 \
  --chunk_size=3 \
  --sink_size=1 \
  --denoising_step_list="1000,750,500,250" \
  --ema_decay=0.99 \
  --ema_start_step=200 \
  --num_denoise_steps=4 \
  --shift=5.0 \
  --dm_loss_weight=1.0 \
  --fake_score_lr=2e-05 \
  --fake_score_rank=512 \
  --dfake_gen_update_ratio=5 \
  --real_guidance_scale=5.0 \
  --transformer_path=$CAUSAL_MODEL_PT \
  --teacher_lora_path=$TEACHER_LORA_PATH \
  --action_pretrain_path=$ACTION_CHECKPOINT_PATH \
  --teacher_action_pretrain_path=$TEACHER_ACTION_CHECKPOINT_PATH \
  --lora_path=$BASE_LORA_PATH \
