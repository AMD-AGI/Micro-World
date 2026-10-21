# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

"""Streaming wrapper for the HummingbirdXT TurboVAED decoder (lossy; decode only).

TurboVAED is non-causal (decoder_causal=false) and has no Wan VAE feat_cache.
Output length is always 4*(T-1)+1. For streaming chunk decode, prepend the last
latent frame of the previous chunk as left context (4 frames -> 13 frames, drop
the first) to match the 12 pixel frames per chunk from the original causal VAE.

The weights directory must contain config.json and diffusion_pytorch_model.safetensors,
e.g. from https://huggingface.co/amd/HummingbirdXT/tree/main/vae/wan22_v1_tiling_16_12
"""
import json
import os

import torch
from safetensors.torch import load_file

from .turbo_vaed_wan22 import AutoencoderKLTurboVAED


class LightVaeDecoder:
    def __init__(self, weights_dir: str, z_dim: int, scale, device, dtype):
        weights_dir = os.path.abspath(weights_dir)
        cfg_path = os.path.join(weights_dir, "config.json")
        weights_path = os.path.join(weights_dir, "diffusion_pytorch_model.safetensors")
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"light_vae config not found: {cfg_path}")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"light_vae weights not found: {weights_path}")

        cfg = json.load(open(cfg_path))
        model = AutoencoderKLTurboVAED.from_config(cfg)
        sd = load_file(weights_path)
        model.decoder.load_state_dict(
            {k: v for k, v in sd.items() if k in model.decoder.state_dict()})
        self.model = model.to(device, dtype).eval()
        self.dtype = dtype
        self.mean = scale[0].to(device, dtype).view(1, z_dim, 1, 1, 1)
        self.inv_std = scale[1].to(device, dtype).view(1, z_dim, 1, 1, 1)
        self._prev_latent = None
        print(f"[light_vae] TurboVAED decoder loaded: {weights_dir} "
              f"(non-causal, +1 latent left context per chunk)")

    def clear_cache(self):
        self._prev_latent = None

    def set_context(self, latents: torch.Tensor):
        self._prev_latent = latents[:, :, -1:].to(self.dtype).clone()

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        z = latents.to(self.dtype)
        if self._prev_latent is None:
            drop = 0
        else:
            z = torch.cat([self._prev_latent, z], dim=2)
            drop = 1
        self._prev_latent = z[:, :, -1:].clone()
        out = self.model.decode(z / self.inv_std + self.mean, return_dict=False)[0]
        return out[:, :, drop:].clamp_(-1, 1)


def attach_light_vae(vae, light_vae: str, device, dtype):
    z_dim = vae.model.z_dim
    vae.light_decoder = LightVaeDecoder(light_vae, z_dim, vae.scale, device, dtype)
    return vae
