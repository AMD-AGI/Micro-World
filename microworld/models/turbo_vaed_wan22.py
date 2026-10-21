# Copyright 2024 The Lightricks team and The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import os.path
# This code is modified based on autoencoder_kl_ltx.py. We thank the original authors for their work.

from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
from diffusers import AutoencoderKLWan
from diffusers.models.activations import get_activation
from diffusers.models.embeddings import PixArtAlphaCombinedTimestepSizeEmbeddings
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import RMSNorm
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.utils import BaseOutput
from timm.models.vision_transformer import Mlp
from dataclasses import dataclass

@dataclass
class DecoderOutput(BaseOutput):
    r"""
    Output of decoding method.

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
            The decoded output sample from the last layer of the model.
    """

    sample: torch.Tensor
    commit_loss: Optional[torch.FloatTensor] = None


class TurboVAEDConv2dSplitUpsampler(nn.Module):
    def __init__(
            self,
            in_channels: int,
            kernel_size: Union[int, Tuple[int, int]] = 3,
            stride: Union[int, Tuple[int, int]] = 1,
            upscale_factor: int = 1,
            padding_mode: str = "zeros",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.upscale_factor = upscale_factor

        out_channels = in_channels

        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)

        height_pad = self.kernel_size[0] // 2
        width_pad = self.kernel_size[1] // 2
        padding = (height_pad, width_pad)

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=padding,
            padding_mode=padding_mode,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = torch.nn.functional.pixel_shuffle(hidden_states, self.stride[0])

        return hidden_states


class TurboVAEDConv2dUpsampler(nn.Module):
    def __init__(
            self,
            in_channels: int,
            kernel_size: Union[int, Tuple[int, int]] = 3,
            stride: Union[int, Tuple[int, int]] = 1,
            upscale_factor: int = 1,
            padding_mode: str = "zeros",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.upscale_factor = upscale_factor

        out_channels = in_channels

        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)

        height_pad = self.kernel_size[0] // 2
        width_pad = self.kernel_size[1] // 2
        padding = (height_pad, width_pad)

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=padding,
            padding_mode=padding_mode,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, channels, time_steps, height, width = hidden_states.shape
        # import pdb;pdb.set_trace()
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(batch_size * time_steps, channels, height, width)

        hidden_states = self.conv(hidden_states)
        hidden_states = torch.nn.functional.pixel_shuffle(hidden_states, self.stride[0])
        _, _, output_height, output_width = hidden_states.shape
        hidden_states = hidden_states.reshape(batch_size, time_steps, -1, output_height, output_width)
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4)

        return hidden_states


class TurboVAEDCausalConv3d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int, int]] = 3,
            stride: Union[int, Tuple[int, int, int]] = 1,
            dilation: Union[int, Tuple[int, int, int]] = 1,
            groups: int = 1,
            padding_mode: str = "zeros",
            is_causal: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_causal = is_causal
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)

        dilation = dilation if isinstance(dilation, tuple) else (dilation, 1, 1)
        stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        height_pad = self.kernel_size[1] // 2
        width_pad = self.kernel_size[2] // 2
        padding = (0, height_pad, width_pad)

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            padding=padding,
            padding_mode=padding_mode,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        time_kernel_size = self.kernel_size[0]

        if self.is_causal:
            if time_kernel_size > 1:
                pad_left = hidden_states[:, :, :1, :, :].repeat((1, 1, time_kernel_size - 1, 1, 1))
                hidden_states = torch.cat([pad_left, hidden_states], dim=2)
        else:
            if time_kernel_size > 1:
                pad_left = hidden_states[:, :, :1, :, :].repeat((1, 1, (time_kernel_size - 1) // 2, 1, 1))
                pad_right = hidden_states[:, :, -1:, :, :].repeat((1, 1, (time_kernel_size - 1) // 2, 1, 1))
                hidden_states = torch.cat([pad_left, hidden_states, pad_right], dim=2)

        hidden_states = self.conv(hidden_states)
        return hidden_states


class TurboVAEDCausalDepthwiseSeperableConv3d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int, int]] = 3,
            stride: Union[int, Tuple[int, int, int]] = 1,
            dilation: Union[int, Tuple[int, int, int]] = 1,
            padding_mode: str = "zeros",
            is_causal: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_causal = is_causal
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, 1, 1)

        # Calculate padding for height and width dimensions
        height_pad = self.kernel_size[1] // 2
        width_pad = self.kernel_size[2] // 2
        self.padding = (0, height_pad, width_pad)

        # Depthwise Convolution
        self.depthwise_conv = nn.Conv3d(
            in_channels,
            in_channels,
            self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            groups=in_channels,  # Each input channel is convolved separately
            padding=self.padding,
            padding_mode=padding_mode,
        )

        # Pointwise Convolution
        self.pointwise_conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=1,  # 1x1x1 convolution to mix channels
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        time_kernel_size = self.kernel_size[0]
        if time_kernel_size > 1:
            pad_count = (time_kernel_size - 1) // 2
            pad_left = hidden_states[:, :, :1, :, :].repeat((1, 1, pad_count, 1, 1))
            pad_right = hidden_states[:, :, -1:, :, :].repeat((1, 1, pad_count, 1, 1))
            hidden_states = torch.cat([pad_left, hidden_states, pad_right], dim=2)

        # Apply depthwise convolution
        hidden_states = self.depthwise_conv(hidden_states)
        # Apply pointwise convolution
        hidden_states = self.pointwise_conv(hidden_states)

        return hidden_states


class TurboVAEDResnetBlock3d(nn.Module):
    r"""
    A 3D ResNet block used in the TurboVAED model.

    Args:
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`, *optional*):
            Number of output channels. If None, defaults to `in_channels`.
        dropout (`float`, defaults to `0.0`):
            Dropout rate.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        elementwise_affine (`bool`, defaults to `False`):
            Whether to enable elementwise affinity in the normalization layers.
        non_linearity (`str`, defaults to `"swish"`):
            Activation function to use.
        conv_shortcut (bool, defaults to `False`):
            Whether or not to use a convolution shortcut.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            dropout: float = 0.0,
            eps: float = 1e-6,
            elementwise_affine: bool = False,
            non_linearity: str = "swish",
            is_causal: bool = True,
            inject_noise: bool = False,
            timestep_conditioning: bool = False,
            is_upsampler_modified: bool = False,
            is_dw_conv: bool = False,
            dw_kernel_size: int = 3,
    ) -> None:
        super().__init__()

        out_channels = out_channels or in_channels

        self.nonlinearity = get_activation(non_linearity)

        self.conv_operation = TurboVAEDCausalConv3d if not is_dw_conv else TurboVAEDCausalDepthwiseSeperableConv3d
        self.kernel_size = 3 if not is_dw_conv else dw_kernel_size

        self.is_upsampler_modified = is_upsampler_modified
        self.replace_nonlinearity = get_activation("relu")

        self.norm1 = RMSNorm(in_channels, eps=1e-8, elementwise_affine=elementwise_affine)
        self.conv1 = self.conv_operation(
            in_channels=in_channels, out_channels=out_channels, kernel_size=self.kernel_size, is_causal=is_causal
        )

        self.norm2 = RMSNorm(out_channels, eps=1e-8, elementwise_affine=elementwise_affine)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = self.conv_operation(
            in_channels=out_channels, out_channels=out_channels, kernel_size=self.kernel_size, is_causal=is_causal
        )

        self.norm3 = None
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.norm3 = nn.LayerNorm(in_channels, eps=eps, elementwise_affine=True, bias=True)
            self.conv_shortcut = self.conv_operation(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, is_causal=is_causal
            )

        self.per_channel_scale1 = None
        self.per_channel_scale2 = None
        if inject_noise:
            self.per_channel_scale1 = nn.Parameter(torch.zeros(in_channels, 1, 1))
            self.per_channel_scale2 = nn.Parameter(torch.zeros(in_channels, 1, 1))

        self.scale_shift_table = None
        if timestep_conditioning:
            self.scale_shift_table = nn.Parameter(torch.randn(4, in_channels) / in_channels ** 0.5)

    def forward(
            self, inputs: torch.Tensor, temb: Optional[torch.Tensor] = None, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        hidden_states = inputs

        hidden_states = self.norm1(hidden_states.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)

        if self.scale_shift_table is not None:
            temb = temb.unflatten(1, (4, -1)) + self.scale_shift_table[None, ..., None, None, None]
            shift_1, scale_1, shift_2, scale_2 = temb.unbind(dim=1)
            hidden_states = hidden_states * (1 + scale_1) + shift_1

        if self.is_upsampler_modified:
            hidden_states = self.replace_nonlinearity(hidden_states)
        else:
            hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if self.per_channel_scale1 is not None:
            spatial_shape = hidden_states.shape[-2:]
            spatial_noise = torch.randn(
                spatial_shape, generator=generator, device=hidden_states.device, dtype=hidden_states.dtype
            )[None]
            hidden_states = hidden_states + (spatial_noise * self.per_channel_scale1)[None, :, None, ...]

        hidden_states = self.norm2(hidden_states.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)

        if self.scale_shift_table is not None:
            hidden_states = hidden_states * (1 + scale_2) + shift_2

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.per_channel_scale2 is not None:
            spatial_shape = hidden_states.shape[-2:]
            spatial_noise = torch.randn(
                spatial_shape, generator=generator, device=hidden_states.device, dtype=hidden_states.dtype
            )[None]
            hidden_states = hidden_states + (spatial_noise * self.per_channel_scale2)[None, :, None, ...]

        if self.norm3 is not None:
            inputs = self.norm3(inputs.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)

        if self.conv_shortcut is not None:
            inputs = self.conv_shortcut(inputs)

        hidden_states = hidden_states + inputs
        return hidden_states


class TurboVAEDDownsampler3d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: Union[int, Tuple[int, int, int]] = 1,
            is_causal: bool = True,
            padding_mode: str = "zeros",
    ) -> None:
        super().__init__()

        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.group_size = (in_channels * stride[0] * stride[1] * stride[2]) // out_channels

        out_channels = out_channels // (self.stride[0] * self.stride[1] * self.stride[2])

        self.conv = TurboVAEDCausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            is_causal=is_causal,
            padding_mode=padding_mode,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = torch.cat([hidden_states[:, :, : self.stride[0] - 1], hidden_states], dim=2)

        residual = (
            hidden_states.unflatten(4, (-1, self.stride[2]))
            .unflatten(3, (-1, self.stride[1]))
            .unflatten(2, (-1, self.stride[0]))
        )
        residual = residual.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(1, 4)
        residual = residual.unflatten(1, (-1, self.group_size))
        residual = residual.mean(dim=2)

        hidden_states = self.conv(hidden_states)
        hidden_states = (
            hidden_states.unflatten(4, (-1, self.stride[2]))
            .unflatten(3, (-1, self.stride[1]))
            .unflatten(2, (-1, self.stride[0]))
        )
        hidden_states = hidden_states.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(1, 4)
        hidden_states = hidden_states + residual

        return hidden_states


class TurboVAEDUpsampler3d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            stride: Union[int, Tuple[int, int, int]] = 1,
            is_causal: bool = True,
            residual: bool = False,
            upscale_factor: int = 1,
            padding_mode: str = "zeros",
            is_video_dc_ae: bool = False,
    ) -> None:
        super().__init__()

        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.residual = residual
        self.upscale_factor = upscale_factor
        self.is_video_dc_ae = is_video_dc_ae

        out_channels = (in_channels * stride[0] * stride[1] * stride[2]) // upscale_factor

        self.conv = TurboVAEDCausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            is_causal=is_causal,
            padding_mode=padding_mode,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        hidden_states = self.conv(hidden_states)

        # step 1：temporal upsampling
        hidden_states = hidden_states.reshape(batch_size, -1, self.stride[0], num_frames, height, width)
        hidden_states = hidden_states.permute(0, 1, 3, 2, 4, 5)
        hidden_states = hidden_states.reshape(batch_size, -1, num_frames * self.stride[0], height, width)

        # step 2：spatial(2D pixel shuffle)
        upsampled_frames = num_frames * self.stride[0]
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4)
        hidden_states = hidden_states.reshape(batch_size * upsampled_frames, -1, height, width)
        hidden_states = torch.nn.functional.pixel_shuffle(hidden_states, self.stride[1])  # assume stride[1] == stride[2]

        # step 3：reshape
        _, c, h, w = hidden_states.shape
        hidden_states = hidden_states.reshape(batch_size, upsampled_frames, c, h, w)
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]

        # step4：slice
        if not self.is_video_dc_ae:
            hidden_states = hidden_states[:, :, self.stride[0] - 1:]

        return hidden_states


# Adapted from diffusers.models.autoencoders.autoencoder_kl_cogvideox.CogVideoMidBlock3d
class TurboVAEDMidBlock3d(nn.Module):
    r"""
    A middle block used in the TurboVAED model.

    Args:
        in_channels (`int`):
            Number of input channels.
        num_layers (`int`, defaults to `1`):
            Number of resnet layers.
        dropout (`float`, defaults to `0.0`):
            Dropout rate.
        resnet_eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        resnet_act_fn (`str`, defaults to `"swish"`):
            Activation function to use.
        is_causal (`bool`, defaults to `True`):
            Whether this layer behaves causally (future frames depend only on past frames) or not.
    """

    _supports_gradient_checkpointing = True

    def __init__(
            self,
            in_channels: int,
            num_layers: int = 1,
            dropout: float = 0.0,
            resnet_eps: float = 1e-6,
            resnet_act_fn: str = "swish",
            is_causal: bool = True,
            inject_noise: bool = False,
            timestep_conditioning: bool = False,
            is_dw_conv: bool = False,
            dw_kernel_size: int = 3,
    ) -> None:
        super().__init__()

        self.time_embedder = None
        if timestep_conditioning:
            self.time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(in_channels * 4, 0)

        resnets = []
        for _ in range(num_layers):
            resnets.append(
                TurboVAEDResnetBlock3d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout=dropout,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    is_causal=is_causal,
                    inject_noise=inject_noise,
                    timestep_conditioning=timestep_conditioning,
                    is_dw_conv=is_dw_conv,
                    dw_kernel_size=dw_kernel_size
                )
            )
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(
            self,
            hidden_states: torch.Tensor,
            temb: Optional[torch.Tensor] = None,
            generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        r"""Forward method of the `LTXMidBlock3D` class."""

        if self.time_embedder is not None:
            temb = self.time_embedder(
                timestep=temb.flatten(),
                resolution=None,
                aspect_ratio=None,
                batch_size=hidden_states.size(0),
                hidden_dtype=hidden_states.dtype,
            )
            temb = temb.view(hidden_states.size(0), -1, 1, 1, 1)

        for i, resnet in enumerate(self.resnets):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb, generator)
            else:
                hidden_states = resnet(hidden_states, temb, generator)

        return hidden_states


class TurboVAEDUpBlock3d(nn.Module):
    r"""
    Up block used in the TurboVAED model.

    Args:
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`, *optional*):
            Number of output channels. If None, defaults to `in_channels`.
        num_layers (`int`, defaults to `1`):
            Number of resnet layers.
        dropout (`float`, defaults to `0.0`):
            Dropout rate.
        resnet_eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        resnet_act_fn (`str`, defaults to `"swish"`):
            Activation function to use.
        spatio_temporal_scale (`bool`, defaults to `True`):
            Whether or not to use a downsampling layer. If not used, output dimension would be same as input dimension.
            Whether or not to downsample across temporal dimension.
        is_causal (`bool`, defaults to `True`):
            Whether this layer behaves causally (future frames depend only on past frames) or not.
    """

    _supports_gradient_checkpointing = True

    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            num_layers: int = 1,
            dropout: float = 0.0,
            resnet_eps: float = 1e-6,
            resnet_act_fn: str = "swish",
            spatio_temporal_scale: bool = True,
            is_causal: bool = True,
            inject_noise: bool = False,
            timestep_conditioning: bool = False,
            upsample_residual: bool = False,
            upscale_factor: int = 1,
            is_dw_conv: bool = False,
            dw_kernel_size: int = 3,
            spatio_only: bool = False,
            is_video_dc_ae: bool = False,
    ):
        super().__init__()

        out_channels = out_channels or in_channels

        self.time_embedder = None
        if timestep_conditioning:
            self.time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(in_channels * 4, 0)

        self.conv_in = None
        if in_channels != out_channels:
            self.conv_in = TurboVAEDResnetBlock3d(
                in_channels=in_channels,
                out_channels=out_channels,
                dropout=dropout,
                eps=resnet_eps,
                non_linearity=resnet_act_fn,
                is_causal=is_causal,
                inject_noise=inject_noise,
                timestep_conditioning=timestep_conditioning,
                is_dw_conv=is_dw_conv,
                dw_kernel_size=dw_kernel_size
            )

        self.upsamplers = None
        if spatio_temporal_scale:
            stride_up = (2, 2, 2) if not spatio_only else (1, 2, 2)
            self.upsamplers = nn.ModuleList(
                [
                    TurboVAEDUpsampler3d(
                        out_channels * upscale_factor,
                        stride=stride_up,
                        is_causal=is_causal,
                        residual=upsample_residual,
                        upscale_factor=upscale_factor,
                        is_video_dc_ae=is_video_dc_ae,
                    )
                ]
            )

        resnets = []
        for _ in range(num_layers):
            resnets.append(
                TurboVAEDResnetBlock3d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    dropout=dropout,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    is_causal=is_causal,
                    inject_noise=inject_noise,
                    timestep_conditioning=timestep_conditioning,
                    is_dw_conv=is_dw_conv,
                    dw_kernel_size=dw_kernel_size,
                    is_upsampler_modified=(spatio_temporal_scale),
                )
            )
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(
            self,
            hidden_states: torch.Tensor,
            temb: Optional[torch.Tensor] = None,
            generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if self.conv_in is not None:
            hidden_states = self.conv_in(hidden_states, temb, generator)

        if self.time_embedder is not None:
            temb = self.time_embedder(
                timestep=temb.flatten(),
                resolution=None,
                aspect_ratio=None,
                batch_size=hidden_states.size(0),
                hidden_dtype=hidden_states.dtype,
            )
            temb = temb.view(hidden_states.size(0), -1, 1, 1, 1)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        for i, resnet in enumerate(self.resnets):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb, generator)
            else:
                hidden_states = resnet(hidden_states, temb, generator)

        return hidden_states

class TurboVAEDDecoder3d(nn.Module):
    r"""
    The `TurboVAEDDecoder3d` layer of a variational autoencoder that decodes its latent representation into an output
    sample.

    Args:
        in_channels (`int`, defaults to 128):
            Number of latent channels.
        out_channels (`int`, defaults to 3):
            Number of output channels.
        block_out_channels (`Tuple[int, ...]`, defaults to `(128, 256, 512, 512)`):
            The number of output channels for each block.
        spatio_temporal_scaling (`Tuple[bool, ...], defaults to `(True, True, True, False)`:
            Whether a block should contain spatio-temporal upscaling layers or not.
        layers_per_block (`Tuple[int, ...]`, defaults to `(4, 3, 3, 3, 4)`):
            The number of layers per block.
        patch_size (`int`, defaults to `4`):
            The size of spatial patches.
        patch_size_t (`int`, defaults to `1`):
            The size of temporal patches.
        resnet_norm_eps (`float`, defaults to `1e-6`):
            Epsilon value for ResNet normalization layers.
        is_causal (`bool`, defaults to `False`):
            Whether this layer behaves causally (future frames depend only on past frames) or not.
        timestep_conditioning (`bool`, defaults to `False`):
            Whether to condition the model on timesteps.
    """

    def __init__(
            self,
            in_channels: int = 128,
            out_channels: int = 3,
            block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
            spatio_temporal_scaling: Tuple[bool, ...] = (True, True, True, False),
            layers_per_block: Tuple[int, ...] = (4, 3, 3, 3, 4),
            patch_size: int = 4,
            patch_size_t: int = 1,
            resnet_norm_eps: float = 1e-6,
            is_causal: bool = False,
            inject_noise: Tuple[bool, ...] = (False, False, False, False),
            timestep_conditioning: bool = False,
            upsample_residual: Tuple[bool, ...] = (False, False, False, False),
            upsample_factor: Tuple[bool, ...] = (1, 1, 1, 1),
            decoder_is_dw_conv: Tuple[bool, ...] = (False, False, False, False, False),
            decoder_dw_kernel_size: int = 3,
            spatio_only: Tuple[bool, ...] = (False, False, False, False),
            upsampling: bool = False,
            is_video_dc_ae: bool = False,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.out_channels = out_channels

        self.upsampling = upsampling

        block_out_channels = tuple(reversed(block_out_channels))
        spatio_temporal_scaling = tuple(reversed(spatio_temporal_scaling))
        layers_per_block = tuple(reversed(layers_per_block))
        inject_noise = tuple(reversed(inject_noise))
        upsample_residual = tuple(reversed(upsample_residual))
        upsample_factor = tuple(reversed(upsample_factor))
        decoder_is_dw_conv = tuple(reversed(decoder_is_dw_conv))
        spatio_only = tuple(reversed(spatio_only))
        output_channel = block_out_channels[0]

        self.conv_in = TurboVAEDCausalConv3d(
            in_channels=in_channels, out_channels=output_channel, kernel_size=3, stride=1, is_causal=is_causal
        )

        self.mid_block = TurboVAEDMidBlock3d(
            in_channels=output_channel,
            num_layers=layers_per_block[0],
            resnet_eps=resnet_norm_eps,
            is_causal=is_causal,
            inject_noise=inject_noise[0],
            timestep_conditioning=timestep_conditioning,
            is_dw_conv=decoder_is_dw_conv[0],
            dw_kernel_size=decoder_dw_kernel_size,
        )

        # up blocks
        num_block_out_channels = len(block_out_channels)
        self.up_blocks = nn.ModuleList([])
        for i in range(num_block_out_channels):
            input_channel = output_channel // upsample_factor[i]
            output_channel = block_out_channels[i] // upsample_factor[i]
            # print(f'for block {i, spatio_temporal_scaling[i], spatio_only[i]}')
            up_block = TurboVAEDUpBlock3d(
                in_channels=input_channel,
                out_channels=output_channel,
                num_layers=layers_per_block[i + 1],
                resnet_eps=resnet_norm_eps,
                spatio_temporal_scale=spatio_temporal_scaling[i],
                is_causal=is_causal,
                inject_noise=inject_noise[i + 1],
                timestep_conditioning=timestep_conditioning,
                upsample_residual=upsample_residual[i],
                upscale_factor=upsample_factor[i],
                is_dw_conv=decoder_is_dw_conv[i + 1],
                dw_kernel_size=decoder_dw_kernel_size,
                spatio_only=spatio_only[i],
                is_video_dc_ae=is_video_dc_ae,
            )

            self.up_blocks.append(up_block)
        if self.patch_size >= 2:
            self.norm_up_1 = RMSNorm(output_channel, eps=1e-8, elementwise_affine=False)
            self.upsampler2d_1 = TurboVAEDConv2dSplitUpsampler(
                in_channels=output_channel,
                kernel_size=3,
                stride=(2, 2),
            )
            output_channel = output_channel // (2 * 2)
        
        if self.patch_size >= 4:
            self.norm_up_2 = RMSNorm(output_channel, eps=1e-8, elementwise_affine=False)
            self.upsampler2d_2 = TurboVAEDConv2dUpsampler(
                in_channels=output_channel,
                kernel_size=3,
                stride=(2, 2),
            )
            output_channel = output_channel // (2 * 2)

        # out
        if self.patch_size == 1:
            self.norm_out = RMSNorm(output_channel, eps=1e-8, elementwise_affine=False)
        
        self.conv_act = nn.SiLU()

        self.conv_out = TurboVAEDCausalConv3d(
            in_channels=output_channel, out_channels=self.out_channels, kernel_size=3, stride=1, is_causal=is_causal
        )

        # timestep embedding
        self.time_embedder = None
        self.scale_shift_table = None
        if timestep_conditioning:
            self.time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(output_channel * 2, 0)
            self.scale_shift_table = nn.Parameter(torch.randn(2, output_channel) / output_channel ** 0.5)

        self.gradient_checkpointing = False

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None,
                feature_enabled: bool = False) -> torch.Tensor:
        hidden_states = self.conv_in(hidden_states)
        feature_backup = {}
        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module):
                def create_forward(*inputs):
                    return module(*inputs)

                return create_forward

            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.mid_block), hidden_states, temb
            )

            for up_block in self.up_blocks:
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(up_block), hidden_states, temb)
        else:
            # print('=======forward========', hidden_states.shape)
            hidden_states = self.mid_block(hidden_states, temb)
            # print('=======mid block======', hidden_states.shape)
            if feature_enabled:
                feature_backup["mid_block"] = hidden_states

            for index, up_block in enumerate(self.up_blocks):
                hidden_states = up_block(hidden_states, temb)
                # print('=====upblock=======', index, hidden_states.shape)
                if feature_enabled:
                    feature_backup[f"up_block_{index}"] = hidden_states


        if self.patch_size >= 2:
            hidden_states = self.norm_up_1(hidden_states.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
            hidden_states = self.conv_act(hidden_states)
            hidden_states_array = []
            for t in range(hidden_states.shape[2]):
                h = self.upsampler2d_1(hidden_states[:, :, t, :, :])
                hidden_states_array.append(h)
            hidden_states = torch.stack(hidden_states_array, dim=2)
        
        if self.patch_size >= 4:
            hidden_states = self.norm_up_2(hidden_states.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
            hidden_states = self.conv_act(hidden_states)
            hidden_states = self.upsampler2d_2(hidden_states)

        if self.patch_size == 1:
            hidden_states = self.norm_out(hidden_states.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        else:
            variance = hidden_states.pow(2).mean(1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + 1e-8)

        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        if feature_enabled:
            return hidden_states, feature_backup
        else:
            return hidden_states


class AutoencoderKLTurboVAED(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    r"""
    A VAE model with KL loss for encoding images into latents and decoding latent representations into images. Used in
    [LTX](https://huggingface.co/Lightricks/LTX-Video).

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Args:
        in_channels (`int`, defaults to `3`):
            Number of input channels.
        out_channels (`int`, defaults to `3`):
            Number of output channels.
        latent_channels (`int`, defaults to `128`):
            Number of latent channels.
        block_out_channels (`Tuple[int, ...]`, defaults to `(128, 256, 512, 512)`):
            The number of output channels for each block.
        spatio_temporal_scaling (`Tuple[bool, ...], defaults to `(True, True, True, False)`:
            Whether a block should contain spatio-temporal downscaling or not.
        layers_per_block (`Tuple[int, ...]`, defaults to `(4, 3, 3, 3, 4)`):
            The number of layers per block.
        patch_size (`int`, defaults to `4`):
            The size of spatial patches.
        patch_size_t (`int`, defaults to `1`):
            The size of temporal patches.
        resnet_norm_eps (`float`, defaults to `1e-6`):
            Epsilon value for ResNet normalization layers.
        scaling_factor (`float`, *optional*, defaults to `1.0`):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
        encoder_causal (`bool`, defaults to `True`):
            Whether the encoder should behave causally (future frames depend only on past frames) or not.
        decoder_causal (`bool`, defaults to `False`):
            Whether the decoder should behave causally (future frames depend only on past frames) or not.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            latent_channels: int = 128,
            decoder_block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
            layers_per_block: Tuple[int, ...] = (4, 3, 3, 3, 4),
            decoder_layers_per_block: Tuple[int, ...] = (4, 3, 3, 3, 4),
            spatio_temporal_scaling: Tuple[bool, ...] = (True, True, True, False),
            decoder_spatio_temporal_scaling: Tuple[bool, ...] = (True, True, True, False),
            decoder_inject_noise: Tuple[bool, ...] = (False, False, False, False, False),
            upsample_residual: Tuple[bool, ...] = (False, False, False, False),
            upsample_factor: Tuple[int, ...] = (1, 1, 1, 1),
            timestep_conditioning: bool = False,
            patch_size: int = 4,
            patch_size_t: int = 1,
            resnet_norm_eps: float = 1e-6,
            scaling_factor: float = 1.0,
            decoder_causal: bool = False,
            spatial_compression_ratio: int = None,
            temporal_compression_ratio: int = None,
            decoder_is_dw_conv: Tuple[bool, ...] = (False, False, False, False, False),
            decoder_dw_kernel_size: int = 3,
            decoder_spatio_only: Tuple[bool, ...] = (False, False, False, False),
            is_video_dc_ae: bool = False,
            aligned_feature_projection_mode: Optional[str] = None,
            aligned_feature_projection_dim: Optional[List[Tuple[int, int]]] = None,
            aligned_blks_indices: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        self.decoder = TurboVAEDDecoder3d(
            in_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=decoder_block_out_channels,
            spatio_temporal_scaling=decoder_spatio_temporal_scaling,
            layers_per_block=decoder_layers_per_block,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            resnet_norm_eps=resnet_norm_eps,
            is_causal=decoder_causal,
            timestep_conditioning=timestep_conditioning,
            inject_noise=decoder_inject_noise,
            upsample_residual=upsample_residual,
            upsample_factor=upsample_factor,
            decoder_is_dw_conv=decoder_is_dw_conv,
            decoder_dw_kernel_size=decoder_dw_kernel_size,
            spatio_only=decoder_spatio_only,
            is_video_dc_ae=is_video_dc_ae,
        )

        latents_mean = torch.zeros((latent_channels,), requires_grad=False)
        latents_std = torch.ones((latent_channels,), requires_grad=False)
        self.register_buffer("latents_mean", latents_mean, persistent=True)
        self.register_buffer("latents_std", latents_std, persistent=True)

        self.spatial_compression_ratio = (
            patch_size * 2 ** sum(spatio_temporal_scaling)
            if spatial_compression_ratio is None
            else spatial_compression_ratio
        )
        self.temporal_compression_ratio = (
            patch_size_t * 2 ** sum([(x and (not y)) for x,y in zip(decoder_spatio_temporal_scaling, decoder_spatio_only)])
            if temporal_compression_ratio is None
            else temporal_compression_ratio
        )

        if aligned_feature_projection_mode is not None:
            assert aligned_feature_projection_dim is not None
            assert len(aligned_feature_projection_dim) == len(aligned_blks_indices)
            self.aligned_feature_projection_heads = nn.ModuleList()
            for i, idx in enumerate(aligned_blks_indices):
                student_dim, teacher_dim = aligned_feature_projection_dim[i]
                if aligned_feature_projection_mode == 'fc-1layer':
                    self.aligned_feature_projection_heads.append(
                        nn.Linear(student_dim, teacher_dim)
                    )
                elif aligned_feature_projection_mode == 'mlp-1layer':
                    self.aligned_feature_projection_heads.append(
                        Mlp(
                            in_features=student_dim, hidden_features=teacher_dim, out_features=teacher_dim,
                            act_layer=nn.GELU, drop=0.0
                        )
                    )
                elif aligned_feature_projection_mode == 'mlp-2layer':
                    self.aligned_feature_projection_heads.append(
                        nn.Sequential(
                            Mlp(
                                in_features=student_dim, hidden_features=teacher_dim, out_features=teacher_dim,
                                act_layer=nn.GELU, drop=0.0
                            ),
                            Mlp(
                                in_features=teacher_dim, hidden_features=teacher_dim, out_features=teacher_dim,
                                act_layer=nn.GELU, drop=0.0
                            )
                        )
                    )
                elif aligned_feature_projection_mode == 'conv-2layer':
                    self.aligned_feature_projection_heads.append(
                        nn.Sequential(
                            TurboVAEDCausalConv3d(student_dim, teacher_dim, kernel_size=1, is_causal=False),
                            TurboVAEDCausalConv3d(teacher_dim, teacher_dim, kernel_size=1, is_causal=False)
                        )
                    )
        else:
            self.aligned_feature_projection_heads = None

        # When decoding a batch of video latents at a time, one can save memory by slicing across the batch dimension
        # to perform decoding of a single video latent at a time.
        self.use_slicing = False

        # When decoding spatially large video latents, the memory requirement is very high. By breaking the video latent
        # frames spatially into smaller tiles and performing multiple forward passes for decoding, and then blending the
        # intermediate tiles together, the memory requirement can be lowered.
        self.use_tiling = False

        # When decoding temporally long video latents, the memory requirement is very high. By decoding latent frames
        # at a fixed frame batch size (based on `self.num_latent_frames_batch_sizes`), the memory requirement can be lowered.
        self.use_framewise_encoding = False
        self.use_framewise_decoding = False

        # This can be configured based on the amount of GPU memory available.
        # `16` for sample frames and `2` for latent frames are sensible defaults for consumer GPUs.
        # Setting it to higher values results in higher memory usage.
        self.num_sample_frames_batch_size = 16
        self.num_latent_frames_batch_size = 2

        # The minimal tile height and width for spatial tiling to be used
        self.tile_sample_min_height = 512
        self.tile_sample_min_width = 512
        self.tile_sample_min_num_frames = 16

        # The minimal distance between two spatial tiles
        self.tile_sample_stride_height = 448
        self.tile_sample_stride_width = 448
        self.tile_sample_stride_num_frames = 8

    def enable_tiling(
            self,
            tile_sample_min_height: Optional[int] = None,
            tile_sample_min_width: Optional[int] = None,
            tile_sample_min_num_frames: Optional[int] = None,
            tile_sample_stride_height: Optional[float] = None,
            tile_sample_stride_width: Optional[float] = None,
            tile_sample_stride_num_frames: Optional[float] = None,
    ) -> None:
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.

        Args:
            tile_sample_min_height (`int`, *optional*):
                The minimum height required for a sample to be separated into tiles across the height dimension.
            tile_sample_min_width (`int`, *optional*):
                The minimum width required for a sample to be separated into tiles across the width dimension.
            tile_sample_stride_height (`int`, *optional*):
                The minimum amount of overlap between two consecutive vertical tiles. This is to ensure that there are
                no tiling artifacts produced across the height dimension.
            tile_sample_stride_width (`int`, *optional*):
                The stride between two consecutive horizontal tiles. This is to ensure that there are no tiling
                artifacts produced across the width dimension.
        """
        self.use_tiling = True
        self.tile_sample_min_height = tile_sample_min_height or self.tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_sample_min_num_frames = tile_sample_min_num_frames or self.tile_sample_min_num_frames
        self.tile_sample_stride_height = tile_sample_stride_height or self.tile_sample_stride_height
        self.tile_sample_stride_width = tile_sample_stride_width or self.tile_sample_stride_width
        self.tile_sample_stride_num_frames = tile_sample_stride_num_frames or self.tile_sample_stride_num_frames

    def disable_tiling(self) -> None:
        r"""
        Disable tiled VAE decoding. If `enable_tiling` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_tiling = False

    def enable_slicing(self) -> None:
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_slicing = True

    def disable_slicing(self) -> None:
        r"""
        Disable sliced VAE decoding. If `enable_slicing` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_slicing = False

    @torch.no_grad()
    def _decode(
            self, z: torch.Tensor, temb: Optional[torch.Tensor] = None, return_dict: bool = True,
            feature_enabled: bool = False
    ) -> Union[DecoderOutput, torch.Tensor]:
        batch_size, num_channels, num_frames, height, width = z.shape
        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_min_num_frames = self.tile_sample_min_num_frames // self.temporal_compression_ratio

        # print('framewise decoding', self.use_framewise_decoding, num_frames, self.tile_sample_min_num_frames, self.temporal_compression_ratio, tile_latent_min_num_frames)
        if self.use_framewise_decoding and num_frames > tile_latent_min_num_frames:
            return self._temporal_tiled_decode(z, temb, return_dict=return_dict)

        if self.use_tiling and (width > tile_latent_min_width or height > tile_latent_min_height):
            return self.tiled_decode(z, temb, return_dict=return_dict)
        
        if feature_enabled:
            dec, feature = self.decoder(z, temb, feature_enabled=feature_enabled)
            return dec, feature

        dec = self.decoder(z, temb)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    @apply_forward_hook
    @torch.no_grad()
    def decode(
            self, z: torch.Tensor, temb: Optional[torch.Tensor] = None, return_dict: bool = True,
            feature_enabled: bool = False
    ) -> Union[DecoderOutput, torch.Tensor]:
        """
        Decode a batch of images.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """
        if self.use_slicing and z.shape[0] > 1:
            if temb is not None:
                decoded_slices = [
                    self._decode(z_slice, t_slice).sample for z_slice, t_slice in (z.split(1), temb.split(1))
                ]
            else:
                decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            if feature_enabled:
                decoded, feature = self._decode(z, temb, feature_enabled=feature_enabled)
                return decoded, feature
            decoded = self._decode(z, temb).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
                    y / blend_extent
            )
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[4], b.shape[4], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
                    x / blend_extent
            )
        return b

    def blend_t(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-3], b.shape[-3], blend_extent)
        for x in range(blend_extent):
            b[:, :, x, :, :] = a[:, :, -blend_extent + x, :, :] * (1 - x / blend_extent) + b[:, :, x, :, :] * (
                    x / blend_extent
            )
        return b


    @torch.no_grad()
    def tiled_decode(
            self, z: torch.Tensor, temb: Optional[torch.Tensor], return_dict: bool = True
    ) -> Union[DecoderOutput, torch.Tensor]:
        r"""
        Decode a batch of images using a tiled decoder.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """

        batch_size, num_channels, num_frames, height, width = z.shape
        sample_height = height * self.spatial_compression_ratio
        sample_width = width * self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        blend_height = self.tile_sample_min_height - self.tile_sample_stride_height
        blend_width = self.tile_sample_min_width - self.tile_sample_stride_width

        # Split z into overlapping tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, height, tile_latent_stride_height):
            row = []
            for j in range(0, width, tile_latent_stride_width):
                time = self.decoder(z[:, :, :, i: i + tile_latent_min_height, j: j + tile_latent_min_width], temb)

                row.append(time)
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(tile[:, :, :, : self.tile_sample_stride_height, : self.tile_sample_stride_width])
            result_rows.append(torch.cat(result_row, dim=4))

        dec = torch.cat(result_rows, dim=3)[:, :, :, :sample_height, :sample_width]

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    @torch.no_grad()
    def _temporal_tiled_decode(
            self, z: torch.Tensor, temb: Optional[torch.Tensor], return_dict: bool = True
    ) -> Union[DecoderOutput, torch.Tensor]:
        batch_size, num_channels, num_frames, height, width = z.shape
        num_sample_frames = (num_frames - 1) * self.temporal_compression_ratio + 1

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_min_num_frames = self.tile_sample_min_num_frames // self.temporal_compression_ratio
        tile_latent_stride_num_frames = self.tile_sample_stride_num_frames // self.temporal_compression_ratio
        blend_num_frames = self.tile_sample_min_num_frames - self.tile_sample_stride_num_frames
        # print(tile_latent_min_num_frames, tile_latent_stride_num_frames, blend_num_frames)
        row = []

        for i in range(0, num_frames, tile_latent_stride_num_frames):
            tile = z[:, :, i: i + tile_latent_min_num_frames + 1, :, :]
            if self.use_tiling and (tile.shape[-1] > tile_latent_min_width or tile.shape[-2] > tile_latent_min_height):
                decoded = self.tiled_decode(tile, temb, return_dict=True).sample
            else:
                decoded = self.decoder(tile, temb)
            if i > 0:
                decoded = decoded[:, :, :-1, :, :]
            # print('for tiled results', tile_latent_stride_num_frames, i, tile.shape, decoded.shape, tile_latent_min_num_frames, self.temporal_compression_ratio)
            row.append(decoded)
            if i + tile_latent_stride_num_frames + 1 == num_frames:
                break

        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_num_frames)
                tile = tile[:, :, : self.tile_sample_stride_num_frames, :, :]
                result_row.append(tile)
            else:
                result_row.append(tile[:, :, : self.tile_sample_stride_num_frames + 1, :, :])

        dec = torch.cat(result_row, dim=2)[:, :, :num_sample_frames]

        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)

    @torch.no_grad()
    def forward(
            self,
            sample: torch.Tensor,
            temb: Optional[torch.Tensor] = None,
            sample_posterior: bool = False,
            return_dict: bool = True,
            generator: Optional[torch.Generator] = None,
    ) -> Union[torch.Tensor, torch.Tensor]:
        dec = self.decode(sample, temb)
        if not return_dict:
            return (dec.sample,)
        return dec
    
    def get_last_layer(self):
        if hasattr(self.decoder.conv_out, "conv"):
            return self.decoder.conv_out.conv.weight
        else:
            return self.decoder.conv_out.weight

    
    

def load_json_to_dict(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data_dict = json.load(file)
        return data_dict
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
def load_vae(dirname):
    model_config_path = os.path.join(dirname, 'config.json')
    model = AutoencoderKLTurboVAED.from_config(
        config=load_json_to_dict(model_config_path)
    )
    student_model_path = os.path.join(dirname, 'diffusion_pytorch_model.safetensors')
    assert os.path.isfile(model_config_path) and os.path.isfile(student_model_path), model_config_path + '|' + student_model_path
    if 'safetensor' in student_model_path:
        from safetensors.torch import load_file
        checkpoint = load_file(student_model_path)
    else:
        checkpoint = torch.load(student_model_path, map_location="cpu")

    if 'state_dict' in checkpoint:
        checkpoint = {kk if 'decoder' not in kk else kk.replace('decoder.', ''): vv for kk, vv in
                      checkpoint['state_dict']['gen_model'].items()}
    # print(len(checkpoint), len(model.decoder.state_dict()))
    # checkpoint = {k[8:]:v for k,v in checkpoint.items() if 'decoder' in k}
    # print('=======', len(set(checkpoint.keys()) & set(model.decoder.state_dict().keys())),
    #       len(set(checkpoint.keys())), len(set(model.decoder.state_dict().keys())))
    checkpoint = {k: v for k, v in checkpoint.items() if k in model.decoder.state_dict()}
    model.decoder.load_state_dict(checkpoint)
    return model


def post_process_for_wan22_dit_latent(vae, output_latent):
    latents_mean = [-0.2289,-0.0052,-0.1323,-0.2339,-0.2799,0.0174,0.1838,0.1557,-0.1382,0.0542,0.2813,0.0891,0.157,-0.0098,0.0375,-0.1825,-0.2246,-0.1207,-0.0698,0.5109,0.2665,-0.2108,-0.2158,0.2502,-0.2055,-0.0322,0.1109,0.1567,-0.0729,0.0899,-0.2799,-0.123,-0.0313,-0.1649,0.0117,0.0723,-0.2839,-0.2083,-0.052,0.3748,0.0152,0.1957,0.1433,-0.2944,0.3573,-0.0548,-0.1681,-0.0667]
    latents_std = [0.4765,1.0364,0.4514,1.1677,0.5313,0.499,0.4818,0.5013,0.8158,1.0344,0.5894,1.0901,0.6885,0.6165,0.8454,0.4978,0.5759,0.3523,0.7135,0.6804,0.5833,1.4146,0.8986,0.5659,0.7069,0.5338,0.4889,0.4917,0.4069,0.4999,0.6866,0.4093,0.5709,0.6065,0.6415,0.4944,0.5726,1.2042,0.5458,1.6887,0.3971,1.06,0.3943,0.5537,0.5444,0.4089,0.7468,0.7744]

    z_dim = 48
    latents = output_latent.to(vae.dtype)
    # print('org latents', latents.min(), latents.max(), latents.mean(), latents.std())
    latents_mean = (
        torch.tensor(latents_mean)
        .view(1, z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(latents_std).view(1, z_dim, 1, 1, 1).to(
        latents.device, latents.dtype
    )
    latents = latents / latents_std + latents_mean
    # print('processed latents', latents.min(), latents.max(), latents.mean(), latents.std())
    return latents

def ours_decode(pipe, vae, latents):
    latents = post_process_for_wan22_dit_latent(vae, latents)
    with torch.no_grad():
        video = vae.decode(latents, return_dict=False)[0].detach()
    #video = pipe.video_processor.postprocess_video(video, output_type='np')
    video = (video / 2 + 0.5).clamp(0, 1)
    return video


def load_vae_for_videox_fun(vae_dir):
    vae = load_vae(vae_dir).to('cuda')
    vae.latent_channels = vae.config.latent_channels

    vae.spacial_compression_ratio = vae.config.scale_factor_spatial
    vae.temporal_compression_ratio = vae.config.scale_factor_temporal
    vae.config.temporal_compression_ratio = vae.config.scale_factor_temporal

    return vae
