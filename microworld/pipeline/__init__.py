# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
from .pipeline_wan_fun import WanFunPipeline
from .pipeline_wan_fun_inpaint import WanFunInpaintPipeline
from .pipeline_wan_action_t2w import WanActionT2WPipeline
from .pipeline_wan_action_i2w import WanActionI2WPipeline
from .pipeline_wan_ti2v import Wan2_2TI2VPipeline
from .pipeline_wan_action_ti2w import Wan2_2ActionTI2WPipeline
from .pipeline_wan_causal import WanCausalPipeline
from .pipeline_wan_causal_dmd import WanCausalDMDPipeline


WanPipeline = WanFunPipeline
WanI2VPipeline = WanFunInpaintPipeline