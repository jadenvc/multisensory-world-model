# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import argparse
import os
import sys
from pathlib import Path
import wandb

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler, Subset
from tqdm import tqdm
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from safetensors.torch import save_file, load_file
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from distributed.fsdp import shard_model, apply_ac
from distributed.util import (
    _configure_model, 
    init_distributed, 
    dist_mean, 
    dist_max
)
from einops import rearrange
from modules.utils import (
    load_transformer,
)
from utils import (
    init_logger, 
    logger, 
    get_mesh_id, 
    sample_timestep_id,
    data_seq_to_patch,
    warmup_constant_lambda,
    FlowMatchScheduler
)

from dataset import MultiLatentLeRobotDataset
import gc


def pad_and_collate(batch):
    """Collate samples with variable frame counts by padding along the F dimension (dim 1)."""
    max_f = max(sample['latents'].shape[1] for sample in batch)

    latents_list, actions_list, actions_mask_list, text_emb_list = [], [], [], []
    latents_mask_list = []

    for sample in batch:
        f = sample['latents'].shape[1]
        pad = max_f - f

        lat = sample['latents']  # [C, F, H, W]
        act = sample['actions']  # [C, F, N, 1]
        amsk = sample['actions_mask']  # [C, F, N, 1]

        if pad > 0:
            lat = torch.nn.functional.pad(lat, (0, 0, 0, 0, 0, pad))
            act = torch.nn.functional.pad(act, (0, 0, 0, 0, 0, pad))
            amsk = torch.nn.functional.pad(amsk, (0, 0, 0, 0, 0, pad))

        frame_mask = torch.zeros(max_f, dtype=torch.bool)
        frame_mask[:f] = True

        latents_list.append(lat)
        actions_list.append(act)
        actions_mask_list.append(amsk)
        text_emb_list.append(sample['text_emb'])
        latents_mask_list.append(frame_mask)

    return {
        'latents': torch.stack(latents_list),
        'actions': torch.stack(actions_list),
        'actions_mask': torch.stack(actions_mask_list),
        'text_emb': torch.stack(text_emb_list),
        'latents_mask': torch.stack(latents_mask_list),  # [B, F]
    }


def _wandb_run_name_from_dataset(config):
    dataset_path = getattr(config, "dataset_path", None)
    if isinstance(dataset_path, (list, tuple)):
        dataset_names = [Path(str(path)).name for path in dataset_path]
        return "+".join(name for name in dataset_names if name)
    if dataset_path:
        return Path(str(dataset_path)).name
    return getattr(config, "config_name", "train")


def _dataset_paths_from_config(config):
    dataset_path = getattr(config, "dataset_path", None)
    if isinstance(dataset_path, (list, tuple)):
        return [str(path) for path in dataset_path]
    if dataset_path:
        return [str(dataset_path)]
    return []


def _quat_xyzw_to_uva_rot6d(quat):
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    x, y, z, w = quat.unbind(dim=-1)

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    row0 = torch.stack([
        1.0 - 2.0 * (yy + zz),
        2.0 * (xy - wz),
        2.0 * (xz + wy),
    ], dim=-1)
    row1 = torch.stack([
        2.0 * (xy + wz),
        1.0 - 2.0 * (xx + zz),
        2.0 * (yz - wx),
    ], dim=-1)
    return torch.cat([row0, row1], dim=-1)


class Trainer:
    def __init__(self, config):
        self.wandb = None
        if config.enable_wandb and config.rank == 0:
            wandb_login_kwargs = {"key": os.environ["WANDB_API_KEY"]}
            wandb_base_url = os.getenv("WANDB_BASE_URL")
            if wandb_base_url:
                wandb_login_kwargs["host"] = wandb_base_url
            wandb.login(**wandb_login_kwargs)
            wandb_init_kwargs = {
                "project": os.getenv("WANDB_PROJECT", "lingbot"),
                "config": config,
                "mode": "online",
                "name": os.getenv("WANDB_RUN_NAME", _wandb_run_name_from_dataset(config)),
            }
            wandb_team_name = os.getenv("WANDB_TEAM_NAME")
            if wandb_team_name:
                wandb_init_kwargs["entity"] = wandb_team_name
            self.wandb = wandb
            self.wandb.init(**wandb_init_kwargs)
            logger.info("WandB logging enabled")
        self.step = 0
        self.config = config
        self.device = torch.device(f"cuda:{config.local_rank}")
        self.dtype = config.param_dtype
        self.patch_size = config.patch_size

        # Load models
        logger.info("Loading models...")

        # Load and shard transformer with FSDP
        logger.info("Loading transformer...")

        if hasattr(config, 'resume_from') and config.resume_from:
            transformer_path = os.path.join(config.resume_from, 'transformer')
            if config.rank == 0:
                logger.info(f"Resuming from checkpoint: {transformer_path}")
        else:
            transformer_path = os.path.join(config.wan22_pretrained_model_name_or_path, 'transformer')

        self.transformer = load_transformer(
            transformer_path,
            torch_dtype=torch.float32,
            torch_device='cpu',
        )

        logger.info("Setting up activation checkpointing ...")
        apply_ac(self.transformer)

        logger.info("Setting up FSDP...")
        shard_fn = shard_model
        self.transformer = _configure_model(
            model=self.transformer,
            shard_fn=shard_fn,
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=False,
        )
        self.transformer.train()
        self.transformer.requires_grad_(True)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            [p for p in self.transformer.parameters() if p.requires_grad],
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=1e-8,
            weight_decay=config.weight_decay,
            fused=True,
            foreach=False,
        )

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, 
            lr_lambda=lambda step: warmup_constant_lambda(step, warmup_steps=config.warmup_steps))

        # Setup dataloaders
        logger.info("Setting up datasets...")
        full_dataset = MultiLatentLeRobotDataset(config=config)
        train_dataset = full_dataset
        self.enable_validation = getattr(config, 'enable_validation', False)
        self.val_loader = None
        if self.enable_validation:
            train_indices, val_indices, val_episode_info = full_dataset.get_episode_split_indices(
                validation_fraction=getattr(config, 'validation_fraction', 0.02),
                validation_size=getattr(config, 'validation_size', 128),
                validation_seed=getattr(config, 'validation_seed', 42),
                return_episode_info=True,
            )
            train_dataset = Subset(full_dataset, train_indices)
            val_dataset = Subset(full_dataset, val_indices)
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=config.world_size,
                rank=config.rank,
                shuffle=False,
            ) if config.world_size > 1 else None
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=getattr(config, 'validation_batch_size', config.batch_size),
                shuffle=False,
                num_workers=getattr(config, 'validation_load_worker', config.load_worker),
                sampler=val_sampler,
                collate_fn=pad_and_collate,
            )
            if config.rank == 0:
                logger.info(
                    f"Episode-level validation enabled: train samples={len(train_indices)}, "
                    f"validation samples={len(val_indices)}"
                )
                self._log_validation_episodes(val_episode_info)
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=config.world_size,
            rank=config.rank,
            shuffle=True,
            seed=42
        ) if config.world_size > 1 else None
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=(train_sampler is None),
            num_workers=config.load_worker,
            sampler=train_sampler,
            collate_fn=pad_and_collate,
        )

        self.train_scheduler_latent = FlowMatchScheduler(shift=self.config.snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_latent.set_timesteps(1000, training=True)
        self.train_scheduler_action = FlowMatchScheduler(shift=self.config.action_snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_action.set_timesteps(1000, training=True)

        self.save_dir = Path(config.save_root) / "checkpoints"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
        self.train_loader_iter = None
        # if hasattr(config, 'resume_from') and config.resume_from:
        #     self._load_training_state(config.resume_from)

    def _log_validation_episodes(self, val_episode_info):
        if self.config.rank != 0:
            return

        dataset_paths = _dataset_paths_from_config(self.config)
        episode_log_rows = []
        for episode_info in val_episode_info:
            row = dict(episode_info)
            dataset_id = row["dataset_id"]
            if dataset_id < len(dataset_paths):
                row["dataset_path"] = dataset_paths[dataset_id]
            else:
                row["dataset_path"] = ""
            episode_log_rows.append(row)

        logger.info(f"Validation episodes ({len(episode_log_rows)}):")
        for row in episode_log_rows:
            logger.info(
                "  "
                f"dataset_id={row['dataset_id']} "
                f"episode_index={row['episode_index']} "
                f"num_samples={row['num_samples']} "
                f"sample_range=[{row['first_sample_index']}, {row['last_sample_index']}] "
                f"dataset_path={row['dataset_path']}"
            )

        if not self.config.enable_wandb or self.wandb is None:
            return

        self.wandb.config.update({
            "validation_episodes": episode_log_rows,
            "validation_episode_count": len(episode_log_rows),
        }, allow_val_change=True)
        table = self.wandb.Table(columns=[
            "dataset_id",
            "dataset_path",
            "episode_index",
            "num_samples",
            "first_sample_index",
            "last_sample_index",
        ])
        for row in episode_log_rows:
            table.add_data(
                row["dataset_id"],
                row["dataset_path"],
                row["episode_index"],
                row["num_samples"],
                row["first_sample_index"],
                row["last_sample_index"],
            )
        self.wandb.log({"validation/episodes": table}, step=0)

    def _scheduler_sigmas_for_timesteps(self, scheduler, timesteps, sample):
        timestep_ids = torch.argmin(
            (scheduler.timesteps[:, None].to(timesteps.device) - timesteps.flatten()[None]).abs(),
            dim=0,
        )
        sigma = scheduler.sigmas.to(sample.device, sample.dtype)[timestep_ids]
        return sigma.reshape(timesteps.shape[0], 1, timesteps.shape[1], 1, 1)

    def _format_predictions(self, input_dict, pred):
        latent_pred, action_pred = pred
        action_pred = rearrange(
            action_pred,
            'b (f n) c -> b c f n 1',
            f=input_dict['action_dict']['targets'].shape[-3],
        )
        latent_pred = data_seq_to_patch(
            self.patch_size,
            latent_pred,
            input_dict['latent_dict']['targets'].shape[-3],
            input_dict['latent_dict']['targets'].shape[-2],
            input_dict['latent_dict']['targets'].shape[-1],
            batch_size=latent_pred.shape[0],
        )
        return latent_pred, action_pred
    
    def _get_next_batch(self):
        """Get next batch from iterator, reset if epoch is finished."""
        if self.train_loader_iter is None:
            self.train_loader_iter = iter(self.train_loader)
        
        try:
            batch = next(self.train_loader_iter)
        except StopIteration:
            # Reset sampler and iterator when epoch finishes
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(self.train_loader.sampler.epoch + 1)
            self.train_loader_iter = iter(self.train_loader)
            batch = next(self.train_loader_iter)
        
        return batch

    @torch.no_grad()
    def _add_noise(self, latent, train_scheduler, action_mask=False, action_mode=False, noisy_cond_prob=0.):
        B, C, F, H, W = latent.shape

        timestep_ids = sample_timestep_id(batch_size=F, num_train_timesteps=train_scheduler.num_train_timesteps)
        noise = torch.zeros_like(latent).normal_()
        timesteps = train_scheduler.timesteps[timestep_ids].to(device=self.device)
        noisy_latents =train_scheduler.add_noise(latent, noise, timesteps, t_dim=2)
        targets =train_scheduler.training_target(latent, noise, timesteps)

        patch_f, patch_h, patch_w = self.patch_size
        if action_mode:
            patch_f = patch_h = patch_w = 1
        
        latent_grid_id = get_mesh_id(
            latent.shape[-3] // patch_f,  # F
            latent.shape[-2] // patch_h,  # H
            latent.shape[-1] // patch_w,  # W
            t=1 if action_mode else 0,  # 1 for action mode (0 for latent), not used
            f_w=1,
            f_shift=0,
            action=action_mode
        ).to(self.device)  # shape: [4, seq_len]
        latent_grid_id = latent_grid_id[None].repeat(B, 1, 1)

        if torch.rand(1).item() < noisy_cond_prob:
            cond_timestep_ids = sample_timestep_id(
                    batch_size=F,
                    min_timestep_bd=0.5, 
                    max_timestep_bd=1.0, 
                    num_train_timesteps=train_scheduler.num_train_timesteps,
                )
            noise = torch.zeros_like(latent).normal_()
            cond_timesteps = train_scheduler.timesteps[cond_timestep_ids].to(device=self.device)
            latent = train_scheduler.add_noise(latent, noise, cond_timesteps, t_dim=2)
        else:
            cond_timesteps = torch.zeros_like(timesteps)

        if action_mask is not None:
            noisy_latents *= action_mask.float()
            targets *= action_mask.float()
            latent *= action_mask.float()

        return dict(
            timesteps=timesteps[None].repeat(B, 1),
            noisy_latents=noisy_latents,
            targets=targets,
            latent=latent,
            cond_timesteps=cond_timesteps[None].repeat(B, 1),
            grid_id=latent_grid_id,
        )

    @torch.no_grad()
    def _prepare_input_dict(self, batch_dict):
        """Prepare input dict following infer code pattern from wan_va_server.py."""
        # Generate grid_id following infer code (no batch dimension yet)
        # For action mode: get_mesh_id(shape[-3], shape[-2], shape[-1], t=1, f_w=1, f_shift, action=True)
        latent_dict = self._add_noise(
            latent=batch_dict['latents'], 
            train_scheduler=self.train_scheduler_latent, 
            action_mask=None, 
            action_mode=False,
            noisy_cond_prob=0.5)
        
        action_dict = self._add_noise(
            latent=batch_dict['actions'], 
            train_scheduler=self.train_scheduler_action, 
            action_mask=batch_dict['actions_mask'], 
            action_mode=True,
            noisy_cond_prob=0.0)

        latent_dict['text_emb'] = batch_dict['text_emb']
        latent_dict['latents_mask'] = batch_dict['latents_mask']
        action_dict['text_emb'] = batch_dict['text_emb']
        action_dict['actions_mask'] = batch_dict['actions_mask']

        input_dict = {
            'latent_dict': latent_dict,
            'action_dict': action_dict,
            'chunk_size': torch.randint(1, 5, (1,)).item(),
            'window_size': torch.randint(4, 65, (1,)).item(),
        }
        return input_dict

    def convert_input_format(self, input_dict):
        """Convert input dict to match transformer input format if needed."""
        for key, value in input_dict.items():
            input_dict[key] = value.to(self.device)#.to(self.dtype)
        return input_dict

    def compute_loss(self,
        input_dict,
        pred
    ):
        latent_pred, action_pred = self._format_predictions(input_dict, pred)
        Bn, Fn = input_dict['latent_dict']['timesteps'].shape
        latent_loss_weight = self.train_scheduler_latent.training_weight(input_dict['latent_dict']['timesteps'].flatten()).reshape(Bn, Fn)
        action_loss_weight = self.train_scheduler_action.training_weight(input_dict['action_dict']['timesteps'].flatten()).reshape(Bn, Fn)

        # Frame-wise video loss calculation
        latent_loss = F.mse_loss(latent_pred.float(), input_dict['latent_dict']['targets'].float().detach(), reduction='none')
        latent_loss = latent_loss * latent_loss_weight[:, None, :, None, None]
        # Permute to (B, F, H, W, C) and flatten to (B*F, H*W*C)
        latent_loss = latent_loss.permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        latent_loss = latent_loss.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        # Sum per frame and normalize, masking out padded frames
        latent_loss_per_frame = latent_loss.sum(dim=1)  # (B*F,)
        latent_mask_per_frame = torch.ones_like(latent_loss).sum(dim=1)  # (B*F,) spatial count
        latent_frame_mask = input_dict['latent_dict']['latents_mask'].flatten().float()  # (B*F,)
        latent_loss = (latent_loss_per_frame / (latent_mask_per_frame + 1e-6) * latent_frame_mask).sum() / latent_frame_mask.sum().clamp(min=1.0)

        # Frame-wise action loss calculation
        action_loss = F.mse_loss(action_pred.float(), input_dict['action_dict']['targets'].float().detach(), reduction='none')
        action_loss = action_loss * action_loss_weight[:, None, :, None, None]
        action_loss = action_loss * input_dict['action_dict']['actions_mask'].float()
        # Permute to (B, F, H, W, C) and flatten to (B*F, H*W*C)
        action_loss = action_loss.permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        action_mask = input_dict['action_dict']['actions_mask'].float().permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        action_loss = action_loss.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        action_mask = action_mask.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        # Sum per frame and normalize by mask per frame
        action_loss_per_frame = action_loss.sum(dim=1)  # (B*F,)
        action_mask_per_frame = action_mask.sum(dim=1)  # (B*F,)
        action_loss = (action_loss_per_frame / (action_mask_per_frame + 1e-6)).mean()

        return latent_loss / self.gradient_accumulation_steps, action_loss / self.gradient_accumulation_steps

    def compute_validation_metrics(self, input_dict, pred):
        latent_pred, action_pred = self._format_predictions(input_dict, pred)
        latent_dict = input_dict['latent_dict']
        action_dict = input_dict['action_dict']
        action_mask = action_dict['actions_mask'].float()

        latent_frame_mask = latent_dict['latents_mask'].float()  # [B, F]
        latent_denoise_err = F.mse_loss(
            latent_pred.float(),
            latent_dict['targets'].float().detach(),
            reduction='none',
        ).mean(dim=[1, 3, 4])  # [B, F] mean over C, H, W
        latent_denoise_mse = (latent_denoise_err * latent_frame_mask).sum() / latent_frame_mask.sum().clamp(min=1.0)
        action_denoise_error = (
            F.mse_loss(
                action_pred.float(),
                action_dict['targets'].float().detach(),
                reduction='none',
            )
            * action_mask
        )
        action_denoise_mse = action_denoise_error.sum() / action_mask.sum().clamp(min=1.0)

        latent_sigma = self._scheduler_sigmas_for_timesteps(
            self.train_scheduler_latent,
            latent_dict['timesteps'],
            latent_dict['noisy_latents'],
        )
        action_sigma = self._scheduler_sigmas_for_timesteps(
            self.train_scheduler_action,
            action_dict['timesteps'],
            action_dict['noisy_latents'],
        )
        pred_clean_latent = latent_dict['noisy_latents'] - latent_sigma * latent_pred
        pred_clean_action = action_dict['noisy_latents'] - action_sigma * action_pred

        latent_recon_mse = F.mse_loss(
            pred_clean_latent.float(),
            latent_dict['latent'].float().detach(),
        )
        action_diff = (pred_clean_action.float() - action_dict['latent'].float().detach()) * action_mask
        valid_action_steps = action_mask.sum(dim=1).squeeze(-1) > 0
        action_l2_norm = torch.sqrt(action_diff.pow(2).sum(dim=1).squeeze(-1).clamp(min=0.0))
        action_l2_norm = action_l2_norm[valid_action_steps].mean()

        q01 = torch.tensor(self.config.norm_stat['q01'], device=self.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
        q99 = torch.tensor(self.config.norm_stat['q99'], device=self.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
        pred_action_real = (pred_clean_action.float() + 1.0) * 0.5 * (q99 - q01) + q01
        target_action_real = (action_dict['latent'].float() + 1.0) * 0.5 * (q99 - q01) + q01
        action_real_diff = (pred_action_real - target_action_real) * action_mask
        action_l2_real = torch.sqrt(action_real_diff.pow(2).sum(dim=1).squeeze(-1).clamp(min=0.0))
        action_l2_real = action_l2_real[valid_action_steps].mean()

        pred_action_steps = pred_action_real.squeeze(-1).permute(0, 2, 3, 1)
        target_action_steps = target_action_real.squeeze(-1).permute(0, 2, 3, 1)
        action_mask_steps = action_mask.squeeze(-1).permute(0, 2, 3, 1).bool()
        pred_action_pose9 = torch.cat([
            pred_action_steps[..., :3],
            _quat_xyzw_to_uva_rot6d(pred_action_steps[..., 3:7]),
        ], dim=-1)
        target_action_pose9 = torch.cat([
            target_action_steps[..., :3],
            _quat_xyzw_to_uva_rot6d(target_action_steps[..., 3:7]),
        ], dim=-1)
        valid_uva_action_steps = action_mask_steps[..., :7].all(dim=-1)
        uva_action_l2 = torch.sqrt(
            (target_action_pose9 - pred_action_pose9).pow(2).sum(dim=-1).clamp(min=0.0)
        )
        uva_action_l2 = uva_action_l2[valid_uva_action_steps].mean()

        return {
            'validation/latent_denoise_mse': latent_denoise_mse.detach(),
            'validation/latent_recon_mse': latent_recon_mse.detach(),
            'validation/action_denoise_mse': action_denoise_mse.detach(),
            'validation/action_l2_norm': action_l2_norm.detach(),
            'validation/action_l2_real': action_l2_real.detach(),
            'validation/action_l2_uva_pose9': uva_action_l2.detach(),
            'val_action_l2_distances': uva_action_l2.detach(),
        }

    @torch.no_grad()
    def validate(self):
        if not self.enable_validation or self.val_loader is None:
            return {}

        was_training = self.transformer.training
        self.transformer.eval()
        metric_sums = {}
        metric_count = 0
        max_batches = getattr(self.config, 'validation_num_batches', 8)

        for batch_idx, batch in enumerate(self.val_loader):
            if batch_idx >= max_batches:
                break
            batch = self.convert_input_format(batch)
            input_dict = self._prepare_input_dict(batch)
            output = self.transformer(input_dict, train_mode=True)
            metrics = self.compute_validation_metrics(input_dict, output)
            for key, value in metrics.items():
                metric_sums[key] = metric_sums.get(key, torch.zeros_like(value)) + value
            metric_count += 1

        if was_training:
            self.transformer.train()

        if metric_count == 0:
            return {}

        out = {}
        for key, value in metric_sums.items():
            local_mean = value / metric_count
            out[key] = dist_mean(local_mean).detach().cpu().item()
        return out

    def _train_step(self, batch, batch_idx):
        """Train a single batch, returns losses for logging."""
        batch = self.convert_input_format(batch)
        input_dict = self._prepare_input_dict(batch)
        
        should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0
        
        if not should_sync:
            self.transformer.set_requires_gradient_sync(False)
        else:
            self.transformer.set_requires_gradient_sync(True)

        output = self.transformer(input_dict, train_mode=True)
        latent_loss, action_loss = self.compute_loss(input_dict, output)
        loss = latent_loss + action_loss

        loss.backward()

        losses = {'latent_loss': latent_loss.detach(), 'action_loss': action_loss.detach()}
        
        # Only update weights after accumulating gradients
        if should_sync:
            total_norm = torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 2.0)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            
            losses['total_norm'] = total_norm
            losses['should_log'] = True
        else:
            losses['should_log'] = False

        return losses

    def save_checkpoint(self,):
        """Save model checkpoint in the same format as pretrained model."""
        try:
            state_dict = get_model_state_dict(
                self.transformer,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
            state_dict_bf16 = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
            # optim_state = get_optimizer_state_dict(
            #         self.transformer, self.optimizer,
            #         options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            #     )

            # Only rank 0 saves the checkpoint
            if self.config.rank == 0:
                checkpoint_dir = self.save_dir / f"checkpoint_step_{self.step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                # Save transformer in the same format as pretrained model
                transformer_dir = checkpoint_dir / "transformer"
                transformer_dir.mkdir(parents=True, exist_ok=True)

                logger.info(f"Saving transformer to {transformer_dir}")

                # Manually save in diffusers format (outside FSDP context to avoid deadlock)
                # Save model weights
                model_file = transformer_dir / "diffusion_pytorch_model.safetensors"
                save_file(state_dict_bf16, model_file)

                # Save config (copy from original transformer config and update _name_or_path)
                config_file = transformer_dir / "config.json"
                config_dict = dict(self.transformer.config)
                config_dict.pop('_name_or_path', None)
                with open(config_file, 'w') as f:
                    json.dump(config_dict, f, indent=2)

                # # Save optimizer state and training metadata in PyTorch format
                # training_state_path = checkpoint_dir / "training_state.pt"
                # logger.info(f"Saving training state to {training_state_path}")
                # torch.save({
                #     'step': self.step,
                #     'optimizer_state_dict': optim_state,
                #     'config': vars(self.config),
                # }, training_state_path)

                logger.info(f"Checkpoint saved successfully at step {self.step}")

            # Synchronize all processes after saving
            if dist.is_initialized():
                dist.barrier()

        except Exception as e:
            if self.config.rank == 0:
                logger.error(f"Failed to save checkpoint: {e}")
                import traceback
                logger.error(traceback.format_exc())
            # Ensure all processes stay synchronized even on error
            if dist.is_initialized():
                dist.barrier()

    def _load_training_state(self, checkpoint_path):
        """Load training state (optimizer + step) after FSDP and optimizer creation."""
        checkpoint_dir = Path(checkpoint_path)
        training_state_path = checkpoint_dir / "training_state.pt"

        if not training_state_path.exists():
            if self.config.rank == 0:
                logger.warning(f"Training state not found: {training_state_path}, starting from step 0")
            return

        if self.config.rank == 0:
            logger.info(f"Loading training state from {training_state_path}")

        # All ranks load the training state directly
        training_state = torch.load(training_state_path, map_location='cpu', weights_only=False)

        # All ranks load optimizer state (required for FSDP)
        set_optimizer_state_dict(
            self.transformer, self.optimizer,
            optim_state_dict=training_state['optimizer_state_dict'],
            options=StateDictOptions(full_state_dict=True, strict=False)
        )
        self.step = training_state.get('step', 0)

        if self.config.rank == 0:
            logger.info(f"Training state loaded, resuming from step {self.step}")

        # Synchronize all ranks
        if dist.is_initialized():
            dist.barrier()

    def train(self):
        """Main training loop - train by steps instead of epochs."""
        logger.info(f"Starting training for {self.config.num_steps} steps...")
        self.transformer.train()

        progress_bar = tqdm(
            total=self.config.num_steps,
            desc="Training",
            disable=(self.config.rank != 0),
            leave=True,
            dynamic_ncols=True,
            initial=self.step
        )

        self.optimizer.zero_grad()
        accumulated_latent_losses = []
        accumulated_action_losses = []
        step_in_accumulation = 0

        while self.step < self.config.num_steps:
            # Get next batch (handles epoch reset automatically)
            batch = self._get_next_batch()
            
            losses = self._train_step(batch, step_in_accumulation)
            
            # Accumulate losses for logging
            accumulated_latent_losses.append(losses['latent_loss'])
            accumulated_action_losses.append(losses['action_loss'])
            step_in_accumulation += 1

            # Log and checkpoint when optimizer steps
            if losses['should_log']:
                lr = self.lr_scheduler.get_last_lr()[0]

                # Average accumulated losses
                latent_loss_show = dist_mean(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                action_loss_show = dist_mean(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()
                max_latent_loss_show = dist_max(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                max_action_loss_show = dist_max(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()

                # Clear accumulated losses
                accumulated_latent_losses = []
                accumulated_action_losses = []
                step_in_accumulation = 0

                torch.cuda.synchronize()
                if self.step % self.config.gc_interval == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                if self.config.rank == 0:
                    total_norm = losses['total_norm']
                    progress_bar.n += self.gradient_accumulation_steps
                    progress_bar.set_postfix({
                        'latent_loss': f'{latent_loss_show:.4f}',
                        'action_loss': f'{action_loss_show:.4f}',
                        'step': self.step,
                        'grad_norm': f'{total_norm.item():.2f}',
                        'lr': f'{lr:.2e}'
                    })
                    if self.config.enable_wandb:
                        self.wandb.log({
                            'loss_metrics/global_avg_video_loss': latent_loss_show,
                            'loss_metrics/global_avg_action_loss': action_loss_show,
                            'loss_metrics/global_max_video_loss': max_latent_loss_show,
                            'loss_metrics/global_max_action_loss': max_action_loss_show,
                            'grad_norm': total_norm.item(),
                            'lr': lr,
                        }, step=self.step)
                
                self.step += 1

                if (
                    self.enable_validation
                    and self.step % getattr(self.config, 'validation_interval', 500) == 0
                ):
                    val_metrics = self.validate()
                    if self.config.rank == 0 and val_metrics:
                        logger.info(f"Validation at step {self.step}: {val_metrics}")
                        if self.config.enable_wandb:
                            self.wandb.log(val_metrics, step=self.step)
                
                if self.step % self.config.save_interval == 0:
                    if self.config.rank == 0:
                        logger.info(f"Starting save model at step {self.step}")
                    self.save_checkpoint()

            if dist.is_initialized():
                dist.barrier()

        progress_bar.close()
        logger.info("Training completed!")


def run(args):
    """Main entry point."""
    config = VA_CONFIGS[args.config_name]

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    init_distributed(world_size, local_rank, rank)

    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size
    config.config_name = args.config_name

    if args.save_root is not None:
        config.save_root = args.save_root
    config.enable_validation = args.enable_validation
    if args.validation_interval is not None:
        config.validation_interval = args.validation_interval
    if args.validation_num_batches is not None:
        config.validation_num_batches = args.validation_num_batches
    if args.validation_size is not None:
        config.validation_size = args.validation_size
    if args.validation_fraction is not None:
        config.validation_fraction = args.validation_fraction
    if args.validation_batch_size is not None:
        config.validation_batch_size = args.validation_batch_size

    if rank == 0:
        logger.info(f"Using config: {args.config_name}")
        logger.info(f"World size: {world_size}, Local rank: {local_rank}")

    trainer = Trainer(config)
    trainer.train()


def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(description="Train WAN model for robotics")
    parser.add_argument(
        "--config-name",
        type=str,
        default='robotwin_train',
        help="Config name",
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default=None,
        help="Root directory for saving checkpoints",
    )
    parser.add_argument(
        "--enable-validation",
        action="store_true",
        help="Enable lightweight latent/action validation metrics during training.",
    )
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=None,
        help="Optimizer-step interval for validation when --enable-validation is set.",
    )
    parser.add_argument(
        "--validation-num-batches",
        type=int,
        default=None,
        help="Maximum validation batches per validation pass.",
    )
    parser.add_argument(
        "--validation-size",
        type=int,
        default=None,
        help="Maximum number of samples to reserve for validation.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=None,
        help="Fraction of the dataset to reserve for validation.",
    )
    parser.add_argument(
        "--validation-batch-size",
        type=int,
        default=None,
        help="Validation batch size; defaults to training batch size.",
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    init_logger()
    main()
