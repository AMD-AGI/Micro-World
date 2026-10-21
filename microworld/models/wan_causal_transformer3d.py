# Modifications Copyright(C)[2025] Advanced Micro Devices, Inc. All rights reserved.
# Modified from https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import glob
import json
import math
import os
import warnings
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from PIL import Image
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.models.modeling_utils import ModelMixin
from torch import nn
from ..utils import cfg_skip
from .wan_transformer3d import WanTransformer3DModel, attention, WanLayerNorm, WanRMSNorm, Head, MLPProj, rope_params, sinusoidal_embedding_1d, WAN_CROSSATTENTION_CLASSES
from .wan_controlnet_action import ActionModule
from . import fused_ops



def patchify_via_linear(x, patch_size, weight, bias):
    """Numerically exact replacement for `nn.Conv3d(kernel_size=patch_size, stride=patch_size)(x)`
    followed by `.flatten(-3).transpose(1, 2)`, valid whenever kernel_size == stride and there is
    no padding (i.e. the "conv" is really just a non-overlapping patchify + linear projection).

    Returns:
        x: Tensor of shape [B, Tp*Hp*Wp, dim] (already flattened/transposed, ready for the transformer)
        grid_sizes: Tensor([Tp, Hp, Wp])
    """
    pt, ph, pw = patch_size
    B, Cin, T, H, W = x.shape
    Tp, Hp, Wp = T // pt, H // ph, W // pw
    x = x.view(B, Cin, Tp, pt, Hp, ph, Wp, pw)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(B, Tp * Hp * Wp, Cin * pt * ph * pw)
    out_channels = weight.shape[0]
    x = torch.nn.functional.linear(x, weight.reshape(out_channels, -1), bias)
    grid_sizes = torch.tensor([Tp, Hp, Wp], dtype=torch.long)
    return x, grid_sizes


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


class CausalWanSelfAttention(nn.Module):
    # Copied from Lingbot and add training part without kv_cache.
    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.debug_collect_attn_stats = False
        self.debug_max_records = 128
        self.debug_attn_stats = []
        self.debug_save_attn_heatmap = False
        self.debug_heatmap_dir = None
        self.debug_heatmap_limit = 1
        self.debug_heatmap_saved = 0

    @torch.no_grad()
    def _save_attn_heatmap(self, probs_all, current_chunk_tokens, branch):
        if not self.debug_save_attn_heatmap:
            return
        if self.debug_heatmap_saved >= self.debug_heatmap_limit:
            return
        if probs_all.ndim != 3 or probs_all.numel() == 0:
            return

        out_dir = self.debug_heatmap_dir or "./attn_debug"
        os.makedirs(out_dir, exist_ok=True)

        # Save one image per head.
        # probs_all: [num_heads, q_len, k_len]
        num_heads, q_len, k_len = probs_all.shape
        probs_np = probs_all.detach().cpu().numpy()
        split = max(0, k_len - int(current_chunk_tokens))

        for head_idx in range(num_heads):
            prob_np = probs_np[head_idx]
            vis = np.log(prob_np + 1e-8)
            vis_min = float(vis.min())
            vis_max = float(vis.max())
            if vis_max > vis_min:
                vis = (vis - vis_min) / (vis_max - vis_min)
            else:
                vis = np.zeros_like(vis, dtype=np.float32)

            r = np.clip((vis - 0.5) * 2.0, 0.0, 1.0)
            g = 1.0 - np.abs(2.0 * vis - 1.0)
            b = np.clip((0.5 - vis) * 2.0, 0.0, 1.0)
            img = np.stack([r, g, b], axis=-1)

            if split > 0 and split < k_len:
                img[:, split - 1:split + 1, :] = np.array([1.0, 1.0, 1.0], dtype=img.dtype)

            layer_idx = int(getattr(self, "layer_idx", -1))
            file_name = (
                f"attn_{self.debug_heatmap_saved:03d}_{branch}"
                f"_l{layer_idx:02d}_h{head_idx:02d}_q{q_len}_k{k_len}_chunk{int(current_chunk_tokens)}.png"
            )
            Image.fromarray((img * 255.0).astype(np.uint8), mode="RGB").save(os.path.join(out_dir, file_name))
        self.debug_heatmap_saved += 1

    @torch.no_grad()
    def _record_attn_stats(self, q_tokens, k_tokens, current_chunk_tokens, branch):
        if (not self.debug_collect_attn_stats) or current_chunk_tokens <= 0:
            return
        if len(self.debug_attn_stats) >= self.debug_max_records:
            return
        if q_tokens.ndim != 4 or k_tokens.ndim != 4:
            return
        if q_tokens.shape[0] == 0 or q_tokens.shape[2] == 0:
            return

        # Use first sample, and collect all heads.
        # q0/kh0: [L, H, D]
        q0 = q_tokens[0].float()
        k0 = k_tokens[0].float()
        if q0.numel() == 0 or k0.numel() == 0:
            return

        # [H, Lq, D] @ [H, D, Lk] -> [H, Lq, Lk]
        qh = q0.permute(1, 0, 2)
        kh = k0.permute(1, 0, 2)
        logits = torch.matmul(qh, kh.transpose(1, 2)) / math.sqrt(qh.shape[-1])
        probs_all = torch.softmax(logits, dim=-1)
        self._save_attn_heatmap(probs_all, current_chunk_tokens, branch)
        total_k = probs_all.shape[-1]
        intra_start = max(0, total_k - current_chunk_tokens)
        history_end = intra_start

        intra_mean = probs_all[:, :, intra_start:].mean().item() if intra_start < total_k else 0.0
        history_mean = probs_all[:, :, :history_end].mean().item() if history_end > 0 else 0.0

        self.debug_attn_stats.append(
            {
                "branch": branch,
                "layer_idx": int(getattr(self, "layer_idx", -1)),
                "num_heads": int(probs_all.shape[0]),
                "q_tokens": int(probs_all.shape[1]),
                "k_tokens": int(total_k),
                "chunk_tokens": int(current_chunk_tokens),
                "history_tokens": int(history_end),
                "intra_mean": float(intra_mean),
                "history_mean": float(history_mean),
                "intra_over_history": float(intra_mean / (history_mean + 1e-8)),
            }
        )

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        kv_cache=None,
        current_start=0,
        max_attention_size=1_000_000,
        clean_x=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if grid_sizes.dim() == 1:
            grid_sizes = grid_sizes.unsqueeze(0).expand(b, -1)

        frame_seqlen = math.prod(grid_sizes[0][1:]).item()
        max_attention_size = frame_seqlen * self.local_attn_size if self.local_attn_size != -1 else 1_000_000

        # query, key, value function
        def qkv_fn(x):
            if fused_ops.QKV_FUSION:
                q_raw, k_raw, v_raw = fused_ops.fused_qkv(self, x)
                if fused_ops.QKV_FUSION_SELFCHECK:
                    ref_q, ref_k, ref_v = self.q(x), self.k(x), self.v(x)
                    if not (torch.allclose(q_raw.float(), ref_q.float(), rtol=1e-2, atol=1e-2)
                            and torch.allclose(k_raw.float(), ref_k.float(), rtol=1e-2, atol=1e-2)
                            and torch.allclose(v_raw.float(), ref_v.float(), rtol=1e-2, atol=1e-2)):
                        fused_ops._warn_once(
                            "qkv_fusion_selfcheck",
                            "[fused_ops] SELFCHECK MISMATCH fused_qkv(): falling back to eager q/k/v.")
                        q_raw, k_raw, v_raw = ref_q, ref_k, ref_v
            else:
                q_raw, k_raw, v_raw = self.q(x), self.k(x), self.v(x)
            q = self.norm_q(q_raw).view(b, s, n, d)
            k = self.norm_k(k_raw).view(b, s, n, d)
            v = v_raw.view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)
        if clean_x is not None:
            _, clean_k, clean_v = qkv_fn(clean_x)

        #Training
        if kv_cache is None:
            F, H, W = grid_sizes[0].tolist()
            HW = H * W
            chunk_size = getattr(self, "chunk_size", 4)  # match VAE temporal compression ratio by default
            independent_first_frame = getattr(self, "independent_first_frame", False)
            sink_tokens = self.sink_size * HW
            frame_spans = []
            if independent_first_frame and F > 1:
                frame_spans.append((0, 1))
                frame_spans.extend(
                    (f_start, min(f_start + chunk_size, F))
                    for f_start in range(1, F, chunk_size)
                )
            else:
                frame_spans.extend(
                    (f_start, min(f_start + chunk_size, F))
                    for f_start in range(0, F, chunk_size)
                )

            def _slice_mixed(raw_start, raw_end, history_end):
                """
                Build K/V window from clean history + current chunk when clean_x is provided.
                clean_x is used for history tokens [0, history_end), current noisy chunk uses [history_end, ...).
                """
                if clean_x is None:
                    return k[:, raw_start:raw_end], v[:, raw_start:raw_end]

                if raw_end <= history_end:
                    return clean_k[:, raw_start:raw_end], clean_v[:, raw_start:raw_end]
                if raw_start >= history_end:
                    return k[:, raw_start:raw_end], v[:, raw_start:raw_end]

                hist_k = clean_k[:, raw_start:history_end]
                hist_v = clean_v[:, raw_start:history_end]
                cur_k = k[:, history_end:raw_end]
                cur_v = v[:, history_end:raw_end]
                return torch.cat([hist_k, cur_k], dim=1), torch.cat([hist_v, cur_v], dim=1)

            outputs = []
            for f_start, f_end in frame_spans:
                num_new_frames = f_end - f_start
                q_chunk_raw = q[:, f_start * HW:f_end * HW]
                k_end = f_end * HW
                history_end = f_start * HW
                window_start = max(0, k_end - max_attention_size)
                prepend_sink = sink_tokens > 0 and window_start > 0

                if prepend_sink:
                    # Keep sink + most recent local window under max_attention_size budget.
                    local_win_start = max(sink_tokens, k_end - (max_attention_size - sink_tokens))
                    sink_raw_k, sink_v = _slice_mixed(0, sink_tokens, history_end)
                    local_raw_k, local_v = _slice_mixed(local_win_start, k_end, history_end)

                    sink_frame_count = self.sink_size
                    local_frame_count = local_raw_k.shape[1] // HW
                    combined_frames = sink_frame_count + local_frame_count

                    sink_grid = grid_sizes.clone()
                    sink_grid[:, 0] = sink_frame_count
                    roped_sink_k = causal_rope_apply(
                        sink_raw_k, sink_grid, freqs, start_frame=0
                    ).type_as(v)

                    local_grid = grid_sizes.clone()
                    local_grid[:, 0] = local_frame_count
                    roped_local_k = causal_rope_apply(
                        local_raw_k, local_grid, freqs, start_frame=sink_frame_count
                    ).type_as(v)

                    window_k = torch.cat([roped_sink_k, roped_local_k], dim=1)
                    window_v = torch.cat([sink_v, local_v], dim=1)
                    q_start_frame = combined_frames - num_new_frames
                else:
                    raw_window_k, window_v = _slice_mixed(window_start, k_end, history_end)
                    window_frames = raw_window_k.shape[1] // HW
                    window_grid = grid_sizes.clone()
                    window_grid[:, 0] = window_frames
                    window_k = causal_rope_apply(
                        raw_window_k, window_grid, freqs, start_frame=0
                    ).type_as(v)
                    q_start_frame = window_frames - num_new_frames

                q_grid = grid_sizes.clone()
                q_grid[:, 0] = num_new_frames
                roped_q_chunk = causal_rope_apply(
                    q_chunk_raw, q_grid, freqs, start_frame=q_start_frame
                ).type_as(v)
                self._record_attn_stats(
                    roped_q_chunk,
                    window_k,
                    current_chunk_tokens=q_chunk_raw.shape[1],
                    branch="train_no_cache",
                )

                out = attention(roped_q_chunk, window_k, window_v)
                outputs.append(out)

            x = torch.cat(outputs, dim=1)

            x = x.flatten(2)
            x = self.o(x)
            return x

        # Inference with kv_cache — dynamic (window-relative) RoPE.
        # Raw k (no RoPE) is stored in the cache. RoPE is applied at attention
        # time using virtual frame indices relative to the attended KV window,
        # so relative distances are always correct even after eviction.
        num_new_frames = grid_sizes[0][0].item()
        current_end = current_start + q.shape[1]
        sink_tokens = self.sink_size * frame_seqlen
        kv_cache_size = kv_cache["k"].shape[1]
        num_new_tokens = q.shape[1]

        if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
            # Eviction: shift local window left, discard oldest tokens after sink.
            num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
            num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
            kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
            kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
            local_end_index = kv_cache["local_end_index"].item() + current_end - \
                kv_cache["global_end_index"].item() - num_evicted_tokens
            local_start_index = local_end_index - num_new_tokens
            kv_cache["k"][:, local_start_index:local_end_index] = k  # store raw key
            kv_cache["v"][:, local_start_index:local_end_index] = v
        else:
            local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
            local_start_index = local_end_index - num_new_tokens
            kv_cache["k"][:, local_start_index:local_end_index] = k  # store raw key
            kv_cache["v"][:, local_start_index:local_end_index] = v

        # Build the attended KV window with relative RoPE.
        # When sink_size=1 and the local window has slid past the sink, prepend
        # the sink frame and give it virtual frames [0..sink_frames), then place
        # the local window at [sink_frames..combined_frames). This keeps all
        # within-window relative distances correct.
        window_start = max(0, local_end_index - max_attention_size)
        prepend_sink = sink_tokens > 0 and window_start > 0

        if prepend_sink:
            sink_frame_count = self.sink_size
            # Local window gets budget = max_attention_size - sink_tokens
            local_win_start = max(sink_tokens, local_end_index - (max_attention_size - sink_tokens))
            local_win_k = kv_cache["k"][:, local_win_start:local_end_index]
            local_win_v = kv_cache["v"][:, local_win_start:local_end_index]
            local_frame_count = local_win_k.shape[1] // frame_seqlen
            combined_frames = sink_frame_count + local_frame_count

            # Sink: virtual frames 0..sink_frame_count
            sink_grid = grid_sizes.clone()
            sink_grid[:, 0] = sink_frame_count
            roped_sink_k = causal_rope_apply(
                kv_cache["k"][:, :sink_tokens], sink_grid, freqs, start_frame=0).type_as(v)

            # Local window: virtual frames sink_frame_count..combined_frames
            local_grid = grid_sizes.clone()
            local_grid[:, 0] = local_frame_count
            roped_local_k = causal_rope_apply(
                local_win_k, local_grid, freqs, start_frame=sink_frame_count).type_as(v)

            window_k = torch.cat([roped_sink_k, roped_local_k], dim=1)
            window_v = torch.cat([kv_cache["v"][:, :sink_tokens], local_win_v], dim=1)
            q_start_frame = combined_frames - num_new_frames
        else:
            # No sink prepend: contiguous recent history, virtual frames start at 0.
            raw_window_k = kv_cache["k"][:, window_start:local_end_index]
            window_v = kv_cache["v"][:, window_start:local_end_index]
            window_frames = raw_window_k.shape[1] // frame_seqlen
            window_grid = grid_sizes.clone()
            window_grid[:, 0] = window_frames
            window_k = causal_rope_apply(
                raw_window_k, window_grid, freqs, start_frame=0).type_as(v)
            q_start_frame = window_frames - num_new_frames

        roped_query = causal_rope_apply(
            q, grid_sizes, freqs, start_frame=q_start_frame).type_as(v)
        self._record_attn_stats(
            roped_query,
            window_k,
            current_chunk_tokens=q.shape[1],
            branch="infer_cache",
        )
        x = attention(roped_query, window_k, window_v)

        kv_cache["global_end_index"].fill_(current_end)
        kv_cache["local_end_index"].fill_(local_end_index)
        
        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim=dim,
                                                num_heads=num_heads,
                                                qk_norm=qk_norm,
                                                eps=eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def _forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        dtype=torch.float32,
        t=0,
        kv_cache=None,
        current_start=0,
        clean_x=None,
        e_clean=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            kv_cache(dict, optional): KV cache for inference acceleration
            current_start(int): Token offset for RoPE positioning with KV cache
        """
        if e.dim() == 3:
            e = (self.modulation + e).chunk(6, dim=1)
        elif e.dim() == 4:
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
            e = [ei.squeeze(2) for ei in e]
        if e_clean is not None:
            if e_clean.dim() == 3:
                e_clean = (self.modulation + e_clean).chunk(6, dim=1)
            elif e_clean.dim() == 4:
                e_clean = (self.modulation.unsqueeze(0) + e_clean).chunk(6, dim=2)
                e_clean = [ei.squeeze(2) for ei in e_clean]

        # self-attention
        # AdaLN modulate/gated-residual are fused via fused_ops (see that module
        # for the WAN_ADALN_FUSION=eager|script backend switch; default 'eager'
        # is numerically identical to the original inline expr).
        temp_x = fused_ops.modulate(self.norm1(x), e[0], e[1])
        temp_x = temp_x.to(dtype)
        clean_temp_x = None
        if clean_x is not None:
            clean_temp_x = fused_ops.modulate(self.norm1(clean_x), e_clean[0], e_clean[1])
            clean_temp_x = clean_temp_x.to(dtype)

        y = self.self_attn(temp_x, seq_lens, grid_sizes, freqs,
                           kv_cache=kv_cache, current_start=current_start,
                           clean_x=clean_temp_x)
        x = fused_ops.gated_residual(x, y, e[2])
        x = x.to(dtype)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            # cross-attention
            x = x + self.cross_attn(self.norm3(x), context, context_lens, dtype)

            # ffn function
            temp_x = fused_ops.modulate(self.norm2(x), e[3], e[4])
            temp_x = temp_x.to(dtype)

            y = self.ffn(temp_x)
            x = fused_ops.gated_residual(x, y, e[5])
            x = x.to(dtype)
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        if clean_x is not None:
            with torch.no_grad():
                clean_y = self.self_attn(clean_temp_x, seq_lens, grid_sizes, freqs)
                clean_x = fused_ops.gated_residual(clean_x, clean_y, e_clean[2])
                clean_x = clean_x.to(dtype)
                clean_x = cross_attn_ffn(clean_x, context, context_lens, e_clean)
            return x, clean_x
        return x

    def forward(self, x, **kwargs):
        if self.training and self.gradient_checkpointing:
            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}
            x = torch.utils.checkpoint.checkpoint(
                self._forward,
                x,
                **kwargs,
                **ckpt_kwargs,
            )
        else:
            x = self._forward(x, **kwargs)
        return x

class CausalWanTransformer3DModel(WanTransformer3DModel):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    # ignore_for_config = [
    #     'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    # ]
    # _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
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
    ):

        super().__init__()
        # WanTransformer3DModel.__init__ is @register_to_config decorated too, so the bare
        # super() call above re-registers ITS defaults on top of the values this class was
        # constructed with. Register them again, otherwise the config saved alongside a
        # checkpoint describes a different architecture than the weights next to it.
        self.register_to_config(
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

        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 'i2v_cross_attn' if model_type == 'i2v' else 't2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                    qk_norm=qk_norm, cross_attn_norm=cross_attn_norm, eps=eps,
                                    )
            for i in range(num_layers)
        ])
        for layer_idx, block in enumerate(self.blocks):
            block.self_attn.layer_idx = layer_idx
            block.self_attn.num_layers = self.num_layers
                    
        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.d = d
        self.dim = dim
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6))
            ],
            dim=1
        )

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        self.teacache = None
        self.cfg_skip_ratio = None
        self.current_steps = 0
        self.num_inference_steps = None
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

        for block in self.blocks:
            block.enable_gradient_checkpointing()

        print(f"CausalWanTransformer3DModel: Gradient checkpointing enabled.")

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

        for block in self.blocks:
            block.disable_gradient_checkpointing()

        print(f"CausalWanTransformer3DModel: Gradient checkpointing disabled.")


    def forward_action(
        self,
        x,
        mouse_actions,
        keyboard_actions,
        kwargs,
        action_kv_caches=None,
        current_start=0,
        frame_offset=0,
    ):
        new_kwargs = dict(x=x)
        new_kwargs.update(kwargs)
        conditions = self.action_preprocess(
            mouse_actions, keyboard_actions,
            grid_sizes=kwargs["grid_sizes"],
            frame_offset=frame_offset,
        )
        for j, block in enumerate(self.action_blocks):
            if action_kv_caches is not None:
                conditions = block(conditions, kv_cache=action_kv_caches[j],
                                   current_start=current_start, **new_kwargs)
            else:
                conditions = block(conditions, **new_kwargs)
        conditions = torch.unbind(conditions)[:-1]
        return conditions

    @cfg_skip()
    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        action_context_scale=1.0,
        cond_flag=True,
        kv_caches=None,
        action_kv_caches=None,
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
            t=t
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