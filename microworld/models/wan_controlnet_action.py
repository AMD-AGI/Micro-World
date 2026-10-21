# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
# Modifications Copyright(C) [Year of 2025] Advanced Micro Devices, Inc. All rights reserved.
import math
import os
import json
import glob
import random
import types
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import register_to_config
from ..parallel import usp_attn_forward
from .wan_transformer3d import WanTransformer3DModel, WanAttentionBlock, sinusoidal_embedding_1d, calculate_freqs_i

class ActionModule(nn.Module):
    def __init__(self, mouse_dim=2, keyboard_dim=7, action_dim=1536, window_size=3, temporal_ratio=4):
        super().__init__()
        self.mouse_dim = mouse_dim
        self.keyboard_dim = keyboard_dim
        self.action_dim = action_dim
        self.window_size = window_size
        self.ratio = temporal_ratio

        # Layers for mouse movement control
        self.mouse_mlp = nn.Sequential(
            nn.Linear(self.ratio * window_size * mouse_dim, action_dim//2),
            nn.GELU(),
            nn.Linear(action_dim//2, action_dim//2)
        )

        keyboard_embedding_dim = 64
        self.keyboard_embedding = nn.Sequential(
            nn.Linear(keyboard_dim, keyboard_embedding_dim),
            nn.GELU(),
            nn.Linear(keyboard_embedding_dim, keyboard_embedding_dim)
        )

        self.keyboard_mlp = nn.Sequential(
            nn.Linear(self.ratio * window_size * keyboard_embedding_dim, action_dim//2),
            nn.GELU(),
            nn.Linear(action_dim//2, action_dim//2)
        )

        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, action_dim//2),
            nn.GELU(),
            nn.Linear(action_dim//2, action_dim)
        )

    def forward(self, mouse_actions, keyboard_actions, grid_sizes):
        """
        Args:
            mouse_actions (Tensor): Shape [B, rn, mouse_dim]
            keyboard_actions (Tensor): Shape [B, rn, keyboard_dim], here keyboard_dim is 7. Like [1, 0, 1, 0, 0, 0, 0]
            features (Tensor): Shape [B, (n+1)*H*W, feature_dim]
        """
        device = mouse_actions.device
        batch_size, rn, _ = mouse_actions.size() # actually, we got rn+1 here. Because every frame have action label.
        t, h, w = grid_sizes
        ## Fisrt process grouped mouse actions
        # Group mouse actions using sliding window
        p0_mouse_actions =  mouse_actions[:, :1].repeat(1, self.ratio*self.window_size,1).reshape(batch_size, -1) # mouse actions in position 0, repeat it.
        grouped_mouse_actions = [p0_mouse_actions] # for index 0, use p0_mouse_actions as init
        for i in range(1, t):
            #The information in the "0" entry can be ignored as it does not correspond to any frame in the video. Each action entry includes
            start_idx = max(1, self.ratio*(i-self.window_size)+1)
            end_idx = i*self.ratio+1
            current_windows = mouse_actions[:, start_idx:end_idx]
            # Pad to fixed length
            target_len = int(self.ratio * self.window_size)
            pad_len = target_len - current_windows.shape[1]
            if pad_len > 0:
                current_windows = F.pad(current_windows,  (0, 0, pad_len, 0),mode='replicate') #For out-of-range indices, boundary actions are used as padding.
            current_windows = current_windows.reshape(batch_size, -1)
            grouped_mouse_actions.append(current_windows) # length = n
        grouped_mouse_actions = torch.stack(grouped_mouse_actions, dim=1) # Shape [B, n+1, ratio*window_size * mouse_dim]
        grouped_mouse_actions = grouped_mouse_actions.unsqueeze(2).repeat(1, 1, h*w, 1) # Repeat step mentioned in paper.
        
        # Process mouse actions
        grouped_mouse_actions = self.mouse_mlp(grouped_mouse_actions)  # Shape [B, t, hw, feature_dim]
        ## Second, process grouped mouse actions
        # Embed and group keyboard actions
        keyboard_embeddings = self.keyboard_embedding(keyboard_actions)  # Shape [B, rn, feature_dim]

        # #add the position encoding as mentioned in paper.
        positions = torch.arange(keyboard_embeddings.size(1), device=keyboard_embeddings.device)
        keyboard_pos_enc = sinusoidal_embedding_1d(keyboard_embeddings.shape[-1], positions)
        keyboard_pos_enc = keyboard_pos_enc.unsqueeze(0).expand(keyboard_embeddings.size(0), -1, -1)
        keyboard_embeddings += keyboard_pos_enc

        p0_keyboard_embeddings = keyboard_embeddings[:,:1].repeat(1, self.ratio*self.window_size,1).reshape(batch_size, -1) # keyboard actions in position 0, repeat it.
        grouped_keyboard_actions = [p0_keyboard_embeddings] # for idx 0, use  p0_keyboard_embeddings as init.
        for i in range(1, t):
            start_idx = max(1, int(self.ratio * (i - self.window_size) + 1))
            end_idx = int(i * self.ratio + 1)
            current_windows = keyboard_embeddings[:, start_idx:end_idx, :]  # [B, window, C]
            target_len = int(self.ratio * self.window_size)
            
            pad_len = target_len - current_windows.shape[1]
            if pad_len > 0:
                # Pad at the start (left) along the sequence dimension
                current_windows = F.pad(current_windows, (0, 0, pad_len, 0),mode='replicate')  # (left, right, top, bottom) for 2D. For out-of-range indices, boundary actions are used as padding.
            current_windows = current_windows.reshape(batch_size, -1)
            grouped_keyboard_actions.append(current_windows)  # [B, target_len, C]

        grouped_keyboard_actions = torch.stack(grouped_keyboard_actions, dim=1)  # Shape [B, n+1, ratio*window_size,feature_dim]
        grouped_keyboard_actions = grouped_keyboard_actions.unsqueeze(2).repeat(1, 1, h*w, 1) # Repeat step mentioned in paper.
        grouped_keyboard_actions = self.keyboard_mlp(grouped_keyboard_actions)
        grouped_actions = torch.concat([grouped_mouse_actions, grouped_keyboard_actions], dim=-1)
        action_features = self.action_mlp(grouped_actions)
        # Process mouse actions
        # grouped_mouse_actions_feat = grouped_mouse_actions_feat.view(batch_size, n_plus_1, h_w, -1) # Shape [B, (n+1), L, ratio*window_size * mouse_dim]
        # mouse_features = self.mouse_mlp(grouped_mouse_actions_feat)  # Shape [B, n+1, L, feature_dim]
        action_features = action_features.flatten(1,2)
        return action_features

class WanActionAttentionBlock(WanAttentionBlock):
    def __init__(
            self,
            cross_attn_type,
            action_dim, 
            dim,
            ffn_dim,
            num_heads,
            window_size=(-1, -1),
            qk_norm=True,
            cross_attn_norm=False,
            eps=1e-6,
            block_id=0,
    ):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = nn.Linear(action_dim, self.dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        self.after_proj = nn.Linear(self.dim, self.dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(self, c, x, **kwargs):
        if self.block_id == 0:
            c = self.before_proj(c) + x
            all_c = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)
        c = super().forward(c, **kwargs)
        c_skip = self.after_proj(c)
        all_c += [c_skip, c]
        c = torch.stack(all_c)
        return c
    
    
class BaseWanAttentionBlock(WanAttentionBlock):
    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        block_id=None
    ):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
        self.block_id = block_id

    def forward(self, x, hints, context_scale=1.0, **kwargs):
        x = super().forward(x, **kwargs)
        if self.block_id is not None:
            x = x + hints[self.block_id] * context_scale
        return x
    
    
class WanActionControlNetModel(WanTransformer3DModel):
    @register_to_config
    def __init__(self,
                 action_layers=None,
                 action_in_dim=None,
                 action_dim=1536,
                 mouse_dim=2,
                 keyboard_dim=7,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        # model_type = "t2v"   # TODO: Hard code for both preview and official versions.
        super().__init__(model_type, patch_size, text_len, in_dim, dim, ffn_dim, freq_dim, text_dim, out_dim,
                         num_heads, num_layers, window_size, qk_norm, cross_attn_norm, eps)

        self.action_layers = [i for i in range(0, self.num_layers, 2)] if action_layers is None else action_layers
        self.action_in_dim = self.in_dim if action_in_dim is None else action_in_dim

        assert 0 in self.action_layers
        self.action_layers_mapping = {i: n for n, i in enumerate(self.action_layers)}
        if model_type == 'i2v':
            cross_attn_type = 'i2v_cross_attn'
        else:
            cross_attn_type = 't2v_cross_attn'
        # blocks
        self.blocks = nn.ModuleList([
            BaseWanAttentionBlock(cross_attn_type, self.dim, self.ffn_dim, self.num_heads, self.window_size, self.qk_norm,
                                  self.cross_attn_norm, self.eps,
                                  block_id=self.action_layers_mapping[i] if i in self.action_layers else None)
            for i in range(self.num_layers)
        ])

        # action blocks
        self.action_blocks = nn.ModuleList([
            WanActionAttentionBlock(cross_attn_type, action_dim, self.dim, self.ffn_dim, self.num_heads, self.window_size, self.qk_norm,
                                    self.cross_attn_norm, self.eps, block_id=i)
            for i in self.action_layers
        ])

        self.action_preprocess = ActionModule(mouse_dim=mouse_dim, keyboard_dim=keyboard_dim, action_dim=action_dim, window_size=3, temporal_ratio=4)
    
    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

        for block in self.blocks:
            block.enable_gradient_checkpointing()
        for condition_block in self.action_blocks:
            condition_block.enable_gradient_checkpointing()
        print(f"WanActionControlNetModel: Gradient checkpointing enabled.")

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

        for block in self.blocks:
            block.disable_gradient_checkpointing()
        for condition_block in self.action_blocks:
            condition_block.disable_gradient_checkpointing()
        print(f"WanActionControlNetModel: Gradient checkpointing disabled.")

    def enable_multi_gpus_inference(self):
        super().enable_multi_gpus_inference()
        if hasattr(self, "action_blocks"):
            for block in self.action_blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)

    def forward_action(
        self,
        x,
        mouse_actions,
        keyboard_actions,
        kwargs
    ):
        # arguments
        new_kwargs = dict(x=x)
        new_kwargs.update(kwargs)
        new_kwargs.pop("seq_len_padded", None)
        conditions = self.action_preprocess(mouse_actions, keyboard_actions, grid_sizes=kwargs["grid_sizes"])
        if self.sp_world_size > 1:
            seq_len = kwargs["seq_len_padded"]
            if conditions.size(1) < seq_len:
                conditions = torch.cat(
                    [conditions, conditions.new_zeros(conditions.size(0), seq_len - conditions.size(1), conditions.size(2))],
                    dim=1,
                )
            conditions = torch.chunk(conditions, self.sp_world_size, dim=1)[self.sp_world_rank]
        # embeddings
        for block in self.action_blocks:
            conditions = block(conditions, **new_kwargs)
        
        conditions = torch.unbind(conditions)[:-1]
        return conditions

    def forward(
        self,
        x,
        t,
        mouse_actions,
        keyboard_actions,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        action_context_scale=1.0,
        cond_flag=True,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        # if self.model_type == 'i2v':
        #     assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        dtype = x.dtype
        if self.freqs.device != device and torch.device(type="meta") != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = torch.cat([x, y], dim=1)

        # embeddings
        x = self.patch_embedding(x)
        grid_sizes = torch.tensor(x.shape[-3:], dtype=torch.long)

        x = x.flatten(-3).transpose(1, 2)
        seq_lens = torch.tensor([x.size(1)] * x.size(0), dtype=torch.long)
        if self.sp_world_size > 1:
            seq_len = int(math.ceil(seq_len / self.sp_world_size)) * self.sp_world_size
        assert seq_lens.max() <= seq_len
        if self.sp_world_size > 1 and x.size(1) < seq_len:
            x = torch.cat([x, x.new_zeros(x.size(0), seq_len - x.size(1), x.size(2))], dim=1)
        
        fhw = grid_sizes.tolist()
        f_indice = list(range(fhw[0])) # TODO: framepack indice in batch axis canbe different
        fhw = tuple(fhw + f_indice)  # add f_indices to fhw for cache key
        c = self.dim // self.num_heads // 2
        freqs_fhw = calculate_freqs_i(fhw, c, self.freqs, f_indice)
        if freqs_fhw.size(0) < seq_len:
            freqs_fhw = torch.cat(
                [
                    freqs_fhw,
                    freqs_fhw.new_ones(seq_len - freqs_fhw.size(0), *freqs_fhw.shape[1:]),
                ],
                dim=0,
            )
        
        # time embeddings
        with amp.autocast(dtype=torch.float32):
            if t.dim() != 1:
                if t.size(1) < seq_len:
                    pad_size = seq_len - t.size(1)
                    last_elements = t[:, -1:]
                    padding = last_elements.repeat(1, pad_size)
                    t = torch.cat([t, padding], dim=1)
                bt = t.size(0)
                ft = t.flatten()
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim,
                                            ft).unflatten(0, (bt, seq_len)).float())
                e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            else:
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, t).float())
                e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            
        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        if self.sp_world_size > 1:
            x = torch.chunk(x, self.sp_world_size, dim=1)[self.sp_world_rank]
            freqs_fhw = torch.chunk(freqs_fhw, self.sp_world_size, dim=0)[self.sp_world_rank]
            if t.dim() != 1:
                e0 = torch.chunk(e0, self.sp_world_size, dim=1)[self.sp_world_rank]
                e = torch.chunk(e, self.sp_world_size, dim=1)[self.sp_world_rank]

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=freqs_fhw,
            context=context,
            context_lens=context_lens,
            dtype=dtype,
            t=t,
            seq_len_padded=seq_len,
        )

        hints = self.forward_action(x, mouse_actions, keyboard_actions, kwargs)
        # only forward_action consumes the padded length; the main blocks do not accept it
        kwargs.pop("seq_len_padded", None)
        kwargs['hints'] = hints
        kwargs['context_scale'] = action_context_scale

        # TeaCache
        if self.teacache is not None:
            if cond_flag:
                modulated_inp = e0
                skip_flag = self.teacache.cnt < self.teacache.num_skip_start_steps
                if skip_flag:
                    self.should_calc = True
                    self.teacache.accumulated_rel_l1_distance = 0
                else:
                    rel_l1_distance = self.teacache.compute_rel_l1_distance(self.teacache.previous_modulated_input, modulated_inp)
                    self.teacache.accumulated_rel_l1_distance += self.teacache.rescale_func(rel_l1_distance)
                    if self.teacache.accumulated_rel_l1_distance < self.teacache.rel_l1_thresh:
                        self.should_calc = False
                    else:
                        self.should_calc = True
                        self.teacache.accumulated_rel_l1_distance = 0
                self.teacache.previous_modulated_input = modulated_inp
                self.teacache.should_calc = self.should_calc
            else:
                self.should_calc = self.teacache.should_calc

        if self.teacache is not None:
            if not self.should_calc:
                previous_residual = self.teacache.previous_residual_cond if cond_flag else self.teacache.previous_residual_uncond
                x = x + previous_residual.to(x.device)[-x.size()[0]:,]
            else:
                ori_x = x.clone().cpu() if self.teacache.offload else x.clone()

                for block in self.blocks:
                    x = block(x, **kwargs)
                    
                if cond_flag:
                    self.teacache.previous_residual_cond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
                else:
                    self.teacache.previous_residual_uncond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
        else:
            for block in self.blocks:
                x = block(x, **kwargs)

        # head
        x = self.head(x, e, dtype=dtype)

        if self.sp_world_size > 1:
            x = self.all_gather(x, dim=1)
            x = x[:, :seq_lens.max().item()]

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        if self.teacache is not None and cond_flag:
            self.teacache.cnt += 1
            if self.teacache.cnt == self.teacache.num_steps:
                self.teacache.reset()
        return x

    @classmethod
    def from_pretrained(
        cls, pretrained_model_path, subfolder=None, transformer_additional_kwargs={}, new_kwargs={},
        low_cpu_mem_usage=False, torch_dtype=torch.bfloat16
    ):
        '''
        transformer_additional_kwargs is used in original VideoXFun to map args to new names.
        new_kwargs is used to config new control architecture.
        '''
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)
        print(f"loaded 3D transformer's pretrained weights from {pretrained_model_path} ...")

        config_file = os.path.join(pretrained_model_path, 'config.json')
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)

        from diffusers.utils import WEIGHTS_NAME
        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        model_file_safetensors = model_file.replace(".bin", ".safetensors")

        if "dict_mapping" in transformer_additional_kwargs.keys():
            for key in transformer_additional_kwargs["dict_mapping"]:
                transformer_additional_kwargs[transformer_additional_kwargs["dict_mapping"][key]] = config[key]

        if low_cpu_mem_usage:
            try:
                import re

                from diffusers import __version__ as diffusers_version
                from diffusers.models.modeling_utils import \
                    load_model_dict_into_meta
                from diffusers.utils import is_accelerate_available
                if is_accelerate_available():
                    import accelerate
                
                # Instantiate model with empty weights
                with accelerate.init_empty_weights():
                    model = cls.from_config(config, **transformer_additional_kwargs, **new_kwargs)

                param_device = "cpu"
                if os.path.exists(model_file):
                    state_dict = torch.load(model_file, map_location="cpu")
                elif os.path.exists(model_file_safetensors):
                    from safetensors.torch import load_file, safe_open
                    state_dict = load_file(model_file_safetensors)
                else:
                    from safetensors.torch import load_file, safe_open
                    model_files_safetensors = glob.glob(os.path.join(pretrained_model_path, "*.safetensors"))
                    state_dict = {}
                    print(model_files_safetensors)
                    for _model_file_safetensors in model_files_safetensors:
                        _state_dict = load_file(_model_file_safetensors)
                        for key in _state_dict:
                            state_dict[key] = _state_dict[key]

                # Check if action_block keys exist, if not, copy from corresponding block
                has_action_block = any("action_blocks" in key for key in state_dict.keys())
                if not has_action_block:
                    print("action_blocks not found in state_dict, copying from blocks...")
                    action_block_keys = [key for key in model.state_dict().keys() if "action_blocks" in key]
                    for key in action_block_keys:
                        # Extract the action_blocks index: action_blocks.10 -> 10
                        action_idx = int(key.split('.')[1])
                        block_idx = action_idx * 2
                        # Replace action_blocks.{action_idx} with blocks.{block_idx}
                        source_key = key.replace(f"action_blocks.{action_idx}", f"blocks.{block_idx}")
                        if source_key in state_dict:
                            state_dict[key] = state_dict[source_key].clone()
                            print(f"  Copied {source_key} -> {key}")

                if diffusers_version >= "0.33.0":
                    # Diffusers has refactored `load_model_dict_into_meta` since version 0.33.0 in this commit:
                    # https://github.com/huggingface/diffusers/commit/f5929e03060d56063ff34b25a8308833bec7c785.
                    load_model_dict_into_meta(
                        model,
                        state_dict,
                        dtype=torch_dtype,
                        model_name_or_path=pretrained_model_path,
                    )
                else:
                    model._convert_deprecated_attention_blocks(state_dict)
                    # move the params from meta device to cpu
                    missing_keys = set(model.state_dict().keys()) - set(state_dict.keys())
                    if len(missing_keys) > 0:
                        raise ValueError(
                            f"Cannot load {cls} from {pretrained_model_path} because the following keys are"
                            f" missing: \n {', '.join(missing_keys)}. \n Please make sure to pass"
                            " `low_cpu_mem_usage=False` and `device_map=None` if you want to randomly initialize"
                            " those weights or else make sure your checkpoint file is correct."
                        )

                    unexpected_keys = load_model_dict_into_meta(
                        model,
                        state_dict,
                        device=param_device,
                        dtype=torch_dtype,
                        model_name_or_path=pretrained_model_path,
                    )

                    if cls._keys_to_ignore_on_load_unexpected is not None:
                        for pat in cls._keys_to_ignore_on_load_unexpected:
                            unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

                    if len(unexpected_keys) > 0:
                        print(
                            f"Some weights of the model checkpoint were not used when initializing {cls.__name__}: \n {[', '.join(unexpected_keys)]}"
                        )
                
                return model
            except Exception as e:
                print(
                    f"The low_cpu_mem_usage mode is not work because {e}. Use low_cpu_mem_usage=False instead."
                )
        
        model = cls.from_config(config, **transformer_additional_kwargs, **new_kwargs)
        if os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location="cpu")
        elif os.path.exists(model_file_safetensors):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(model_file_safetensors)
        else:
            from safetensors.torch import load_file, safe_open
            model_files_safetensors = glob.glob(os.path.join(pretrained_model_path, "*.safetensors"))
            state_dict = {}
            for _model_file_safetensors in model_files_safetensors:
                _state_dict = load_file(_model_file_safetensors)
                for key in _state_dict:
                    state_dict[key] = _state_dict[key]
        
        # Check if action_block keys exist, if not, copy from corresponding block
        has_action_block = any("action_blocks" in key for key in state_dict.keys())
        if not has_action_block:
            print("action_blocks not found in state_dict, copying from blocks...")
            action_block_keys = [key for key in model.state_dict().keys() if "action_blocks" in key]
            for key in action_block_keys:
                # Extract the action_blocks index: action_blocks.10 -> 10
                action_idx = int(key.split('.')[1])
                block_idx = action_idx * 2
                # Replace action_blocks.{action_idx} with blocks.{block_idx}
                source_key = key.replace(f"action_blocks.{action_idx}", f"blocks.{block_idx}")
                if source_key in state_dict:
                    state_dict[key] = state_dict[source_key].clone()
                    print(f"  Copied {source_key} -> {key}")
        
        if model.state_dict()['patch_embedding.weight'].size() != state_dict['patch_embedding.weight'].size():
            model.state_dict()['patch_embedding.weight'][:, :state_dict['patch_embedding.weight'].size()[1], :, :] = state_dict['patch_embedding.weight']
            model.state_dict()['patch_embedding.weight'][:, state_dict['patch_embedding.weight'].size()[1]:, :, :] = 0
            state_dict['patch_embedding.weight'] = model.state_dict()['patch_embedding.weight']
        
        tmp_state_dict = {} 
        for key in state_dict:
            if key in model.state_dict().keys() and model.state_dict()[key].size() == state_dict[key].size():
                tmp_state_dict[key] = state_dict[key]
            else:
                print(key, "Size don't match, skip")
                
        state_dict = tmp_state_dict

        m, u = model.load_state_dict(state_dict, strict=False)
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
        print(m)
        
        params = [p.numel() if "." in n else 0 for n, p in model.named_parameters()]
        print(f"### All Parameters: {sum(params) / 1e6} M")

        params = [p.numel() if "attn1." in n else 0 for n, p in model.named_parameters()]
        print(f"### attn1 Parameters: {sum(params) / 1e6} M")
        
        model = model.to(torch_dtype)
        return model