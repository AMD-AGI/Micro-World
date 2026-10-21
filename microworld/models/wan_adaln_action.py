# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
# Modifications Copyright(C) [Year of 2025] Advanced Micro Devices, Inc. All rights reserved.
import math
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import register_to_config
from einops import rearrange
from .wan_transformer3d import WanTransformer3DModel, WanAttentionBlock, sinusoidal_embedding_1d, calculate_freqs_i

class ActionModule(nn.Module):
    def __init__(self, mouse_dim=2, keyboard_dim=7, action_dim=1536, window_size=3, temporal_ratio=4,
                 max_position_index=None):
        super().__init__()
        self.mouse_dim = mouse_dim
        self.keyboard_dim = keyboard_dim
        self.action_dim = action_dim
        self.window_size = window_size
        self.ratio = temporal_ratio
        # Upper bound for the keyboard positional encoding index. Rolling out past the
        # trained horizon otherwise feeds the model positions it has never seen, so causal
        # models clamp to their training length. None keeps the unbounded 1.0 behaviour.
        self.max_position_index = max_position_index

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

    def forward(self, mouse_actions, keyboard_actions, grid_sizes, frame_offset=0):
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
        grouped_mouse_actions = [] # for index 0, use p0_mouse_actions as init
        for i in range(t):
            #The information in the "0" entry can be ignored as it does not correspond to any frame in the video. Each action entry includes
            global_i = frame_offset + i
            if global_i == 0:
                grouped_mouse_actions.append(p0_mouse_actions)
                continue
            start_idx = max(1, self.ratio*(global_i-self.window_size)+1)
            end_idx = global_i*self.ratio+1
            current_windows = mouse_actions[:, start_idx:end_idx]
            # Pad to fixed length
            target_len = int(self.ratio * self.window_size)
            pad_len = target_len - current_windows.shape[1]
            if pad_len > 0:
                current_windows = F.pad(current_windows,  (0, 0, pad_len, 0),mode='replicate') #For out-of-range indices, boundary actions are used as padding.
            current_windows = current_windows.reshape(batch_size, -1)
            grouped_mouse_actions.append(current_windows) # length = n
        grouped_mouse_actions = torch.stack(grouped_mouse_actions, dim=1) # Shape [B, n+1, ratio*window_size * mouse_dim]
        
        # Process mouse actions
        grouped_mouse_actions = self.mouse_mlp(grouped_mouse_actions)  # Shape [B, t, hw, feature_dim]
        ## Second, process grouped mouse actions
        # Embed and group keyboard actions
        keyboard_embeddings = self.keyboard_embedding(keyboard_actions)  # Shape [B, rn, feature_dim]

        # #add the position encoding as mentioned in paper.
        positions = torch.arange(keyboard_embeddings.size(1), device=keyboard_embeddings.device)
        if self.max_position_index is not None:
            positions = positions.clamp(max=self.max_position_index)

        keyboard_pos_enc = sinusoidal_embedding_1d(keyboard_embeddings.shape[-1], positions)
        keyboard_pos_enc = keyboard_pos_enc.unsqueeze(0).expand(keyboard_embeddings.size(0), -1, -1)
        keyboard_embeddings += keyboard_pos_enc

        p0_keyboard_embeddings = keyboard_embeddings[:,:1].repeat(1, self.ratio*self.window_size,1).reshape(batch_size, -1) # keyboard actions in position 0, repeat it.
        grouped_keyboard_actions = [] # for idx 0, use  p0_keyboard_embeddings as init.
        for i in range(t):
            global_i = frame_offset + i
            if global_i == 0:
                grouped_keyboard_actions.append(p0_keyboard_embeddings)
                continue
            start_idx = max(1, int(self.ratio * (global_i - self.window_size) + 1))
            end_idx = int(global_i * self.ratio + 1)
            current_windows = keyboard_embeddings[:, start_idx:end_idx, :]  # [B, window, C]
            target_len = int(self.ratio * self.window_size)
            
            pad_len = target_len - current_windows.shape[1]
            if pad_len > 0:
                # Pad at the start (left) along the sequence dimension
                current_windows = F.pad(current_windows, (0, 0, pad_len, 0),mode='replicate')  # (left, right, top, bottom) for 2D. For out-of-range indices, boundary actions are used as padding.
            current_windows = current_windows.reshape(batch_size, -1)
            grouped_keyboard_actions.append(current_windows)  # [B, target_len, C]

        grouped_keyboard_actions = torch.stack(grouped_keyboard_actions, dim=1)  # Shape [B, n+1, ratio*window_size,feature_dim]
        grouped_keyboard_actions = self.keyboard_mlp(grouped_keyboard_actions)
        grouped_actions = torch.concat([grouped_mouse_actions, grouped_keyboard_actions], dim=-1)
        action_features = self.action_mlp(grouped_actions)
        # Process mouse actions
        # grouped_mouse_actions_feat = grouped_mouse_actions_feat.view(batch_size, n_plus_1, h_w, -1) # Shape [B, (n+1), L, ratio*window_size * mouse_dim]
        # mouse_features = self.mouse_mlp(grouped_mouse_actions_feat)  # Shape [B, n+1, L, feature_dim]
        # action_features = action_features.flatten(1,2)
        return action_features    
    
class AdaLNWanAttentionBlock(WanAttentionBlock):
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
    ):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
        self.action_adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(action_dim, 6 * dim, bias=True))

    def forward(self, x, **kwargs):
        grid_sizes = kwargs["grid_sizes"]
        action_feat = kwargs.pop("action_feat")
        action_feat = self.action_adaLN_modulation(action_feat)
        action_feat = rearrange(action_feat, "b t (s d)->b t 1 1 s d", s=6)        
        action_feat = action_feat.repeat(1, 1, grid_sizes[1], grid_sizes[2], 1, 1).flatten(1, 3)
        if kwargs["e"].dim() == 3:
            kwargs["e"] = kwargs["e"].unsqueeze(1)
        kwargs["e"] = kwargs["e"] + action_feat
        
        x = super().forward(x, **kwargs)
        return x
    
    
class WanActionAdaLNModel(WanTransformer3DModel):
    @register_to_config
    def __init__(self,
                 mouse_dim=2,
                 keyboard_dim=7,
                 action_dim=1536,
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
                 eps=1e-6,
                 max_position_index=84):
        # model_type = "t2v"   # TODO: Hard code for both preview and official versions.
        super().__init__(model_type, patch_size, text_len, in_dim, dim, ffn_dim, freq_dim, text_dim, out_dim,
                         num_heads, num_layers, window_size, qk_norm, cross_attn_norm, eps)
        self.action_preprocess = ActionModule(mouse_dim=mouse_dim, keyboard_dim=keyboard_dim, action_dim=action_dim, window_size=3, temporal_ratio=4,
                                              max_position_index=max_position_index)
        if model_type == 'i2v':
            cross_attn_type = 'i2v_cross_attn'
        else:
            cross_attn_type = 't2v_cross_attn'
        # blocks
        self.blocks = nn.ModuleList([
            AdaLNWanAttentionBlock(cross_attn_type, action_dim, self.dim, self.ffn_dim, self.num_heads, self.window_size, self.qk_norm,
                                  self.cross_attn_norm, self.eps)
            for i in range(self.num_layers)
        ])
    
    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

        for block in self.blocks:
            block.enable_gradient_checkpointing()
        print(f"WanActionModel: Gradient checkpointing enabled.")

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

        for block in self.blocks:
            block.disable_gradient_checkpointing()
        print(f"WanActionModel: Gradient checkpointing disabled.")

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
        
        # control adaLN
        conditions = self.action_preprocess(mouse_actions, keyboard_actions, grid_sizes=grid_sizes)  # b, t, d
        if self.sp_world_size > 1:
            conditions = conditions.unsqueeze(2).repeat(1, 1, grid_sizes[1] * grid_sizes[2], 1).flatten(1, 2)
            if conditions.size(1) < seq_len:
                conditions = torch.cat(
                    [conditions, conditions.new_zeros(conditions.size(0), seq_len - conditions.size(1), conditions.size(2))],
                    dim=1,
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
        context = torch.stack([
            torch.cat(
                [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ])
        # Keep context dtype aligned with text_embedding weights under mixed precision.
        context = context.to(dtype=next(self.text_embedding.parameters()).dtype)
        context = self.text_embedding(context)

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        if self.sp_world_size > 1:
            x = torch.chunk(x, self.sp_world_size, dim=1)[self.sp_world_rank]
            freqs_fhw = torch.chunk(freqs_fhw, self.sp_world_size, dim=0)[self.sp_world_rank]
            conditions = torch.chunk(conditions, self.sp_world_size, dim=1)[self.sp_world_rank]
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
            t=t  
        )

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
                    x = block(x, action_feat=conditions, **kwargs)
                    
                if cond_flag:
                    self.teacache.previous_residual_cond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
                else:
                    self.teacache.previous_residual_uncond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
        else:
            for block in self.blocks:
                x = block(x, action_feat=conditions, **kwargs)

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