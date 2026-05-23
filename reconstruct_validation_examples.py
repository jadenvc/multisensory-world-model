#!/usr/bin/env python3
"""Reconstruct validation examples from a LingBot-VA checkpoint.

This is an offline diagnostic script. It uses the same episode-level validation
split as training validation, runs one-step denoising reconstruction for held-out
examples, decodes real/predicted latents to video, and plots action predictions
against ground truth.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from diffusers.utils import export_to_video
from diffusers.video_processor import VideoProcessor
from einops import rearrange
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
sys.path.append(str(REPO_ROOT / "wan_va"))

from configs import VA_CONFIGS  # noqa: E402
from dataset import MultiLatentLeRobotDataset  # noqa: E402
from modules.utils import load_transformer, load_vae  # noqa: E402
from utils import FlowMatchScheduler, data_seq_to_patch, get_mesh_id  # noqa: E402


def move_to_device(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def scheduler_sigmas_for_timesteps(scheduler, timesteps, sample):
    timestep_ids = torch.argmin(
        (scheduler.timesteps[:, None].to(timesteps.device) - timesteps.flatten()[None]).abs(),
        dim=0,
    )
    sigma = scheduler.sigmas.to(sample.device, sample.dtype)[timestep_ids]
    return sigma.reshape(timesteps.shape[0], 1, timesteps.shape[1], 1, 1)


@torch.no_grad()
def add_noise(latent, scheduler, timestep_index, patch_size, device, action_mask=None, action_mode=False):
    batch_size, _, num_frames, _, _ = latent.shape
    timesteps = scheduler.timesteps[timestep_index].to(device=device).repeat(num_frames)
    noise = torch.zeros_like(latent).normal_()
    noisy_latents = scheduler.add_noise(latent, noise, timesteps, t_dim=2)
    targets = scheduler.training_target(latent, noise, timesteps)

    patch_f, patch_h, patch_w = patch_size
    if action_mode:
        patch_f = patch_h = patch_w = 1

    grid_id = get_mesh_id(
        latent.shape[-3] // patch_f,
        latent.shape[-2] // patch_h,
        latent.shape[-1] // patch_w,
        t=1 if action_mode else 0,
        f_w=1,
        f_shift=0,
        action=action_mode,
    ).to(device)

    if action_mask is not None:
        mask = action_mask.float()
        noisy_latents *= mask
        targets *= mask
        latent *= mask

    return {
        "timesteps": timesteps[None].repeat(batch_size, 1),
        "noisy_latents": noisy_latents,
        "targets": targets,
        "latent": latent,
        "cond_timesteps": torch.zeros_like(timesteps)[None].repeat(batch_size, 1),
        "grid_id": grid_id[None].repeat(batch_size, 1, 1),
    }


@torch.no_grad()
def prepare_input_dict(batch, config, latent_scheduler, action_scheduler, timestep_index, device):
    latent_dict = add_noise(
        latent=batch["latents"],
        scheduler=latent_scheduler,
        timestep_index=timestep_index,
        patch_size=config.patch_size,
        device=device,
        action_mask=None,
        action_mode=False,
    )
    action_dict = add_noise(
        latent=batch["actions"],
        scheduler=action_scheduler,
        timestep_index=timestep_index,
        patch_size=config.patch_size,
        device=device,
        action_mask=batch["actions_mask"],
        action_mode=True,
    )
    latent_dict["text_emb"] = batch["text_emb"]
    action_dict["text_emb"] = batch["text_emb"]
    action_dict["actions_mask"] = batch["actions_mask"]
    return {
        "latent_dict": latent_dict,
        "action_dict": action_dict,
        "chunk_size": 2,
        "window_size": 16,
    }


def format_predictions(config, input_dict, pred):
    latent_pred, action_pred = pred
    action_pred = rearrange(
        action_pred,
        "b (f n) c -> b c f n 1",
        f=input_dict["action_dict"]["targets"].shape[-3],
    )
    latent_pred = data_seq_to_patch(
        config.patch_size,
        latent_pred,
        input_dict["latent_dict"]["targets"].shape[-3],
        input_dict["latent_dict"]["targets"].shape[-2],
        input_dict["latent_dict"]["targets"].shape[-1],
        batch_size=latent_pred.shape[0],
    )
    return latent_pred, action_pred


def reconstruct_clean(input_dict, latent_pred, action_pred, latent_scheduler, action_scheduler):
    latent_dict = input_dict["latent_dict"]
    action_dict = input_dict["action_dict"]
    latent_sigma = scheduler_sigmas_for_timesteps(
        latent_scheduler,
        latent_dict["timesteps"],
        latent_dict["noisy_latents"],
    )
    action_sigma = scheduler_sigmas_for_timesteps(
        action_scheduler,
        action_dict["timesteps"],
        action_dict["noisy_latents"],
    )
    pred_clean_latent = latent_dict["noisy_latents"] - latent_sigma * latent_pred
    pred_clean_action = action_dict["noisy_latents"] - action_sigma * action_pred
    pred_clean_action *= action_dict["actions_mask"].float()
    return pred_clean_latent, pred_clean_action


@torch.no_grad()
def decode_video(vae, video_processor, latents):
    latents = latents.to(next(vae.parameters()).device, vae.dtype)
    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = (
        1.0
        / torch.tensor(vae.config.latents_std)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents = latents / latents_std + latents_mean
    video = vae.decode(latents, return_dict=False)[0]
    return video_processor.postprocess_video(video, output_type="np")


def unnormalize_actions(config, actions):
    q01 = torch.tensor(config.norm_stat["q01"], device=actions.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
    q99 = torch.tensor(config.norm_stat["q99"], device=actions.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
    return (actions.float() + 1.0) * 0.5 * (q99 - q01) + q01


def plot_actions(config, pred_actions, target_actions, action_mask, output_path, max_channels):
    pred = unnormalize_actions(config, pred_actions)[0, :, :, :, 0].detach().cpu()
    target = unnormalize_actions(config, target_actions)[0, :, :, :, 0].detach().cpu()
    mask = action_mask[0, :, :, :, 0].detach().cpu().bool()
    active_channels = torch.where(mask.any(dim=(1, 2)))[0].tolist()
    active_channels = active_channels[:max_channels]

    if len(active_channels) == 0:
        return

    time = torch.arange(pred.shape[1] * pred.shape[2])
    fig, axes = plt.subplots(len(active_channels), 1, figsize=(12, 2.2 * len(active_channels)), sharex=True)
    if len(active_channels) == 1:
        axes = [axes]

    used_channels = getattr(config, "used_action_channel_ids", [])
    for axis, channel_id in zip(axes, active_channels):
        pred_curve = pred[channel_id].reshape(-1)
        target_curve = target[channel_id].reshape(-1)
        valid = mask[channel_id].reshape(-1)
        label = f"channel {channel_id}"
        if channel_id in used_channels:
            label += f" (used[{used_channels.index(channel_id)}])"
        axis.plot(time[valid], target_curve[valid], label="gt", linewidth=1.5)
        axis.plot(time[valid], pred_curve[valid], label="pred", linewidth=1.2)
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right")

    axes[-1].set_xlabel("action step within chunk")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Reconstruct LingBot validation examples from a checkpoint.")
    parser.add_argument("--config-name", default="umi_wbw_train")
    parser.add_argument(
        "--checkpoint-root",
        default=None,
        help="Checkpoint root containing transformer/. Defaults to config pretrained path.",
    )
    parser.add_argument(
        "--vae-root",
        default=None,
        help="Checkpoint root containing vae/. Defaults to checkpoint-root if present, otherwise config pretrained path.",
    )
    parser.add_argument("--output-dir", default="reconstruction_out")
    parser.add_argument("--num-examples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--validation-size", type=int, default=128)
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--validation-seed", type=int, default=42)
    parser.add_argument("--noise-timestep-index", type=int, default=500)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-action-channels", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = VA_CONFIGS[args.config_name]
    checkpoint_root = Path(args.checkpoint_root or config.wan22_pretrained_model_name_or_path)
    default_vae_root = checkpoint_root if (checkpoint_root / "vae").exists() else Path(config.wan22_pretrained_model_name_or_path)
    vae_root = Path(args.vae_root) if args.vae_root is not None else default_vae_root
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = config.param_dtype

    dataset = MultiLatentLeRobotDataset(config=config)
    _, val_indices = dataset.get_episode_split_indices(
        validation_fraction=args.validation_fraction,
        validation_size=args.validation_size,
        validation_seed=args.validation_seed,
    )
    val_dataset = Subset(dataset, val_indices[: args.num_examples])
    loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    latent_scheduler = FlowMatchScheduler(shift=config.snr_shift, sigma_min=0.0, extra_one_step=True)
    latent_scheduler.set_timesteps(1000, training=True)
    action_scheduler = FlowMatchScheduler(shift=config.action_snr_shift, sigma_min=0.0, extra_one_step=True)
    action_scheduler.set_timesteps(1000, training=True)
    timestep_index = min(max(args.noise_timestep_index, 0), latent_scheduler.timesteps.numel() - 1)

    transformer = load_transformer(
        str(checkpoint_root / "transformer"),
        torch_dtype=dtype,
        torch_device=device,
    )
    transformer.eval().requires_grad_(False)

    vae = load_vae(
        str(vae_root / "vae"),
        torch_dtype=dtype,
        torch_device=device,
    )
    vae.eval().requires_grad_(False)
    video_processor = VideoProcessor(vae_scale_factor=1)

    for example_idx, batch in enumerate(tqdm(loader, desc="Reconstructing")):
        batch = move_to_device(batch, device)
        input_dict = prepare_input_dict(
            batch,
            config,
            latent_scheduler,
            action_scheduler,
            timestep_index,
            device,
        )
        pred = transformer(input_dict, train_mode=True)
        latent_pred, action_pred = format_predictions(config, input_dict, pred)
        pred_clean_latent, pred_clean_action = reconstruct_clean(
            input_dict,
            latent_pred,
            action_pred,
            latent_scheduler,
            action_scheduler,
        )

        latent_mse = F.mse_loss(pred_clean_latent.float(), input_dict["latent_dict"]["latent"].float()).item()
        action_mask = input_dict["action_dict"]["actions_mask"]
        action_mse = (
            (pred_clean_action.float() - input_dict["action_dict"]["latent"].float()).pow(2)
            * action_mask.float()
        ).sum() / action_mask.float().sum().clamp(min=1.0)

        real_video = decode_video(vae, video_processor, input_dict["latent_dict"]["latent"][:1])[0]
        pred_video = decode_video(vae, video_processor, pred_clean_latent[:1])[0]
        export_to_video(real_video, str(output_dir / f"example_{example_idx:03d}_real.mp4"), fps=args.fps)
        export_to_video(pred_video, str(output_dir / f"example_{example_idx:03d}_pred.mp4"), fps=args.fps)

        plot_actions(
            config,
            pred_clean_action,
            input_dict["action_dict"]["latent"],
            action_mask,
            output_dir / f"example_{example_idx:03d}_actions.png",
            args.max_action_channels,
        )

        print(
            f"example {example_idx}: latent_recon_mse={latent_mse:.6f} "
            f"action_recon_mse={action_mse.item():.6f}"
        )


if __name__ == "__main__":
    main()
