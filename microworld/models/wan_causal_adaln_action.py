# Modifications Copyright(C)[2025] Advanced Micro Devices, Inc. All rights reserved.
# Modified from https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import register_to_config
from einops import rearrange

from ..utils import cfg_skip
from . import fused_ops
from .wan_adaln_action import ActionModule
from .wan_causal_transformer3d import CausalWanAttentionBlock, CausalWanTransformer3DModel, patchify_via_linear
from .wan_transformer3d import sinusoidal_embedding_1d


class AdaLNCausalWanAttentionBlock(CausalWanAttentionBlock):
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
        action_feat = kwargs.pop("action_feat", None)
        if action_feat is None:
            x = super().forward(x, **kwargs)
            return x
        action_feat = self.action_adaLN_modulation(action_feat)
        action_feat = rearrange(action_feat, "b t (s d)->b t 1 1 s d", s=6)        
        action_feat = action_feat.repeat(1, 1, grid_sizes[1], grid_sizes[2], 1, 1).flatten(1, 3)
        if kwargs["e"].dim() == 3:
            kwargs["e"] = kwargs["e"].unsqueeze(1)
        kwargs["e"] = kwargs["e"] + action_feat
        if kwargs.get("e_clean", None) is not None:
            if kwargs["e_clean"].dim() == 3:
                kwargs["e_clean"] = kwargs["e_clean"].unsqueeze(1)
            kwargs["e_clean"] = kwargs["e_clean"] + action_feat
        
        x = super().forward(x, **kwargs)
        return x
    
    
class CausalWanActionAdaLNModel(CausalWanTransformer3DModel):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    @register_to_config
    def __init__(
        self,
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
        in_channels=16,
        max_position_index=84,
    ):

        super().__init__(
            model_type=model_type,
            patch_size=patch_size,
            text_len=text_len,
            in_dim=in_dim,
            dim=dim,
            ffn_dim=ffn_dim,
            freq_dim=freq_dim,
            text_dim=text_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            window_size=window_size,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
            in_channels=in_channels,
        )
        # Rollouts routinely run past the trained horizon, so clamp the action positional
        # encoding to the longest index seen during training.
        self.action_preprocess = ActionModule(mouse_dim=mouse_dim, keyboard_dim=keyboard_dim, action_dim=action_dim, window_size=3, temporal_ratio=4,
                                              max_position_index=max_position_index)

        cross_attn_type = 'i2v_cross_attn' if model_type == 'i2v' else 't2v_cross_attn'
        self.blocks = nn.ModuleList([
            AdaLNCausalWanAttentionBlock(cross_attn_type, action_dim, dim, ffn_dim, num_heads,
                                    qk_norm=qk_norm, cross_attn_norm=cross_attn_norm, eps=eps,
                                    )
            for i in range(num_layers)
        ])
        for layer_idx, block in enumerate(self.blocks):
            block.self_attn.layer_idx = layer_idx
            block.self_attn.num_layers = self.num_layers

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

        for block in self.blocks:
            block.enable_gradient_checkpointing()

        print(f"CausalWanActionAdaLNModel: Gradient checkpointing enabled.")

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

        for block in self.blocks:
            block.disable_gradient_checkpointing()

        print(f"CausalWanActionAdaLNModel: Gradient checkpointing disabled.")

    @cfg_skip()
    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        mouse_actions=None,
        keyboard_actions=None,
        clip_fea=None,
        y=None,
        action_context_scale=1.0,
        cond_flag=True,
        kv_caches=None,
        current_start=0,
        frame_offset=0,
        clean_x=None,
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
            y (Dict[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x
            cond_flag (`bool`, *optional*, defaults to True):
                Flag to indicate whether to forward the condition input

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert y is not None
        # params
        device = self.patch_embedding.weight.device
        dtype = x.dtype
        if self.freqs.device != device and torch.device(type="meta") != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = torch.cat([x, y], dim=1)
            if clean_x is not None:
                clean_x = torch.cat([clean_x, y], dim=1)
        # embeddings
        # Equivalent to self.patch_embedding(x).flatten(-3).transpose(1, 2), but avoids MIOpen's
        # slow Im3d2Col fallback on gfx950 by expressing the non-overlapping patchify as a matmul.
        x, grid_sizes = patchify_via_linear(
            x, self.patch_size, self.patch_embedding.weight, self.patch_embedding.bias)

        if clean_x is not None:
            clean_x, _ = patchify_via_linear(
                clean_x, self.patch_size, self.patch_embedding.weight, self.patch_embedding.bias)

        # Created directly on-device: this same tensor is reused as k_lens/q_lens
        # by every transformer layer's self-attention, so building it on GPU
        # from the start avoids one redundant HtoD copy per layer per forward.
        seq_lens = torch.tensor(
            [x.size(1)] * x.size(0), dtype=torch.long, device=x.device)

        # time embeddings
        with amp.autocast(dtype=torch.float32):
            if t.dim() != 1:
                # t is [B, F] (per-frame timesteps), repeat each frame's timestep for H*W spatial tokens
                F_frames = grid_sizes[0].item() if grid_sizes.dim() == 1 else grid_sizes[0, 0].item()
                HW = seq_len // F_frames
                clean_t = torch.zeros_like(t[:, :F_frames])
                clean_ft = clean_t.unsqueeze(-1).expand(-1, -1, HW).reshape(t.size(0), seq_len).flatten()
                t = t[:, :F_frames].unsqueeze(-1).expand(-1, -1, HW).reshape(t.size(0), seq_len)
                bt = t.size(0)
                ft = t.flatten()
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim,
                                            ft).unflatten(0, (bt, seq_len)).float())
                e0 = self.time_projection(e).unflatten(2, (6, self.dim))
                if clean_x is not None:
                    clean_e = self.time_embedding(
                        sinusoidal_embedding_1d(self.freq_dim,
                                                clean_ft).unflatten(0, (bt, seq_len)).float())
                    e_clean = self.time_projection(clean_e).unflatten(2, (6, self.dim))
            else:
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, t).float())
                e0 = self.time_projection(e).unflatten(1, (6, self.dim))
                if clean_x is not None:
                    clean_t = torch.zeros_like(t)
                    clean_e = self.time_embedding(
                        sinusoidal_embedding_1d(self.freq_dim, clean_t).float())
                    e_clean = self.time_projection(clean_e).unflatten(1, (6, self.dim))

            # assert e.dtype == torch.float32 and e0.dtype == torch.float32

            # assert e.dtype == torch.float32 and e0.dtype == torch.float32
            # e0 = e0.to(dtype)
            # e = e.to(dtype)

        # control adaLN
        if mouse_actions is not None and keyboard_actions is not None:
            action_feat = self.action_preprocess(
                mouse_actions,
                keyboard_actions,
                grid_sizes=grid_sizes,
                frame_offset=frame_offset,
            )
            action_feat = action_feat * action_context_scale
        else:
            action_feat = None

        # context
        context_lens = None
        context = fused_ops.embed_context(self, context)

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)
        
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
        
        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            dtype=dtype,
            t=t,
            action_feat=action_feat
        )
        if clean_x is not None:
            kwargs["e_clean"] = e_clean

        # TeaCache
        if self.teacache is not None:
            if not self.should_calc:
                previous_residual = self.teacache.previous_residual_cond if cond_flag else self.teacache.previous_residual_uncond
                x = x + previous_residual.to(x.device)[-x.size()[0]:,]
            else:
                ori_x = x.clone().cpu() if self.teacache.offload else x.clone()

                for i, block in enumerate(self.blocks):
                    if kv_caches is not None:
                        x = block(x, kv_cache=kv_caches[i], current_start=current_start, **kwargs)
                    else:
                        if clean_x is not None:
                            x, clean_x = block(x, clean_x=clean_x, **kwargs)
                        else:
                            x = block(x, **kwargs)

                if cond_flag:
                    self.teacache.previous_residual_cond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
                else:
                    self.teacache.previous_residual_uncond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
        else:
            for i, block in enumerate(self.blocks):
                if kv_caches is not None:
                    x = block(x, kv_cache=kv_caches[i], current_start=current_start, **kwargs)
                else:
                    if clean_x is not None:
                        x, clean_x = block(x, clean_x=clean_x, **kwargs)
                    else:
                        x = block(x, **kwargs)

        # head
        x = self.head(x, e, dtype=dtype)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        if self.teacache is not None and cond_flag:
            self.teacache.cnt += 1
            if self.teacache.cnt == self.teacache.num_steps:
                self.teacache.reset()
        return x
