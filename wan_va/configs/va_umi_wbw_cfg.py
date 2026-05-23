# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .shared_config import va_shared_cfg

va_umi_wbw_cfg = EasyDict(__name__='Config: VA UMI WBW')
va_umi_wbw_cfg.update(va_shared_cfg)

va_umi_wbw_cfg.wan22_pretrained_model_name_or_path = (
    "/store/real/jvclark/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin"
)

va_umi_wbw_cfg.attn_window = 72
va_umi_wbw_cfg.frame_chunk_size = 2
va_umi_wbw_cfg.env_type = 'none'

va_umi_wbw_cfg.height = 256
va_umi_wbw_cfg.width = 256
va_umi_wbw_cfg.action_dim = 30
va_umi_wbw_cfg.action_per_frame = 16
va_umi_wbw_cfg.obs_cam_keys = ['observation.images.camera0']

va_umi_wbw_cfg.guidance_scale = 5
va_umi_wbw_cfg.action_guidance_scale = 1

va_umi_wbw_cfg.num_inference_steps = 25
va_umi_wbw_cfg.video_exec_step = -1
va_umi_wbw_cfg.action_num_inference_steps = 50

va_umi_wbw_cfg.snr_shift = 5.0
va_umi_wbw_cfg.action_snr_shift = 1.0

# Parquet action (8-dim, pre-computed relative to episode frame 0):
#   [0:7] = rel_xyz + rel_quat_xyzw (scipy xyzw, qw at index 6)
#   [7]   = gripper_width_m
# env_type='none': no runtime transform, inverse mapping places:
#   action_aligned[:, 0:7] = rel EEF, action_aligned[:, 28] = gripper.
va_umi_wbw_cfg.used_action_channel_ids = list(range(0, 7)) + [28]
inverse_used_action_channel_ids = [
    len(va_umi_wbw_cfg.used_action_channel_ids)
] * va_umi_wbw_cfg.action_dim
for i, j in enumerate(va_umi_wbw_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_umi_wbw_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_umi_wbw_cfg.action_norm_method = 'quantiles'
va_umi_wbw_cfg.norm_stat = {
    "q01": [
        -0.6019223153591156,
        -0.4839091670513153,
        -0.4446462419629097,
        -0.43614304542541504,
        -0.18932639434933662,
        -0.2929276701807976,
        0.8068272161483765,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.04059896819293499,
        0.0,
    ],
    "q99": [
        0.15569971427321366,
        -0.0021110750734806166,
        0.0066349845752119935,
        -0.0022509341128170556,
        0.18181128859519943,
        0.47004713028669354,
        0.9999815821647644,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.08438394203782082,
        1.0,
    ],
}
