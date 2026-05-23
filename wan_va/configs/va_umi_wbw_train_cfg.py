# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os

from easydict import EasyDict

from .va_umi_wbw_cfg import va_umi_wbw_cfg

va_umi_wbw_train_cfg = EasyDict(__name__='Config: VA UMI WBW train')
va_umi_wbw_train_cfg.update(va_umi_wbw_cfg)

va_umi_wbw_train_cfg.wan22_pretrained_model_name_or_path = (
    "/store/real/jvclark/lingbot-va/checkpoints/lingbot-va-base"
)
va_umi_wbw_train_cfg.dataset_path = (
    "/store/real/jvclark/lingbot-va/data/relative_umi_wbw_lerobot_v21"
)
# extract_wan22_latents_lerobot.py writes empty_emb.pt to dataset_path's parent
va_umi_wbw_train_cfg.empty_emb_path = (
    "/store/real/jvclark/lingbot-va/data/empty_emb.pt"
)

va_umi_wbw_train_cfg.enable_wandb = True
va_umi_wbw_train_cfg.load_worker = 16
va_umi_wbw_train_cfg.save_interval = 1000
va_umi_wbw_train_cfg.gc_interval = 50
va_umi_wbw_train_cfg.cfg_prob = 0.1

# Training parameters
va_umi_wbw_train_cfg.learning_rate = 1e-5
va_umi_wbw_train_cfg.beta1 = 0.9
va_umi_wbw_train_cfg.beta2 = 0.95
va_umi_wbw_train_cfg.weight_decay = 0.1
va_umi_wbw_train_cfg.warmup_steps = 10
va_umi_wbw_train_cfg.batch_size = 2
va_umi_wbw_train_cfg.gradient_accumulation_steps = 1
va_umi_wbw_train_cfg.num_steps = 50000
