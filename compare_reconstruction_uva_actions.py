#!/usr/bin/env python3
"""Compare UVA and LingBot reconstruction actions on the same trajectory.

This script uses the LingBot validation/reconstruction data path as the source
of truth for LingBot formatting. It reconstructs a dense LingBot action tensor,
converts both dense GT and dense predictions back to UVA's current-relative
pose10d format, and optionally overlays UVA policy predictions on the same
zarr timesteps.
"""

import argparse
import gc
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
UVA_ROOT = Path("/store/real/jvclark/unified_video_action")
sys.path.insert(0, str(UVA_ROOT))
sys.path.insert(0, str(REPO_ROOT / "wan_va"))
sys.path.insert(0, str(REPO_ROOT))

from umi.common.pose_util import mat_to_pose, mat_to_pose10d, pose10d_to_mat, pose_to_mat  # noqa: E402
from unified_video_action.common.pose_repr_util import convert_pose_mat_rep  # noqa: E402


def pose10d_to_axis_angle(action_10d):
    pose_mat = pose10d_to_mat(action_10d[:9])
    pose_6d = mat_to_pose(pose_mat)
    out = np.zeros(7, dtype=np.float32)
    out[:6] = pose_6d
    out[6] = action_10d[9]
    return out


def rotmat_to_rot6d(rot_mat):
    return rot_mat[:, :2].T.reshape(-1)


def current_pose_mat(extractor, episode_idx, timestep):
    start = int(extractor.episode_starts[episode_idx])
    idx = start + int(timestep)
    pos = np.asarray(extractor.data_store["robot0_eef_pos"][idx])
    rot = np.asarray(extractor.data_store["robot0_eef_rot_axis_angle"][idx])
    return pose_to_mat(np.concatenate([pos, rot]))


def episode_zero_pose_mat(extractor, episode_idx):
    start = int(extractor.episode_starts[episode_idx])
    pos = np.asarray(extractor.data_store["robot0_eef_pos"][start])
    rot = np.asarray(extractor.data_store["robot0_eef_rot_axis_angle"][start])
    return pose_to_mat(np.concatenate([pos, rot]))


def lingbot_actions_to_uva_pose10d(action8_frame, ep0_pose_mat, cur_pose_mat):
    """Convert LingBot [rel_xyz_to_ep0, rel_quat_to_ep0, gripper] to UVA pose10d."""
    mats = []
    grippers = []
    ep0_pos = ep0_pose_mat[:3, 3]
    ep0_rot = Rotation.from_matrix(ep0_pose_mat[:3, :3])
    for k in range(action8_frame.shape[1]):
        quat = np.asarray(action8_frame[3:7, k], dtype=np.float64)
        quat_norm = np.linalg.norm(quat)
        if quat_norm < 1e-8:
            quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        else:
            quat = quat / quat_norm
        abs_pose = np.eye(4, dtype=np.float64)
        abs_pose[:3, 3] = ep0_pos + action8_frame[:3, k]
        abs_pose[:3, :3] = (ep0_rot * Rotation.from_quat(quat)).as_matrix()
        mats.append(abs_pose)
        grippers.append(action8_frame[7, k])
    mats = np.asarray(mats)
    rel_current = convert_pose_mat_rep(
        mats,
        base_pose_mat=cur_pose_mat,
        pose_rep="relative",
        backward=False,
    )
    pose9 = mat_to_pose10d(rel_current)
    out = np.zeros((pose9.shape[0], 10), dtype=np.float32)
    out[:, :9] = pose9
    out[:, 9] = np.asarray(grippers, dtype=np.float32)
    return out


def build_uva_obs(dp, device):
    obs = dp["observation"]
    images = torch.from_numpy(obs["camera0_rgb"]).float().permute(0, 3, 1, 2) / 255.0
    images = images.unsqueeze(0).to(device)

    poses_6d = np.concatenate(
        [obs["robot0_eef_pos"], obs["robot0_eef_rot_axis_angle"]],
        axis=-1,
    )
    pose_mats = pose_to_mat(poses_6d)
    current_mat = pose_mats[-1]
    rel_pose_mats = convert_pose_mat_rep(
        pose_mats,
        base_pose_mat=current_mat,
        pose_rep="relative",
        backward=False,
    )
    rel_poses_6d = np.array([mat_to_pose(p) for p in rel_pose_mats])
    rel_rot6d = np.array([rotmat_to_rot6d(p[:3, :3]) for p in rel_pose_mats])

    demo_start_pose = obs.get("robot0_demo_start_pose", None)
    if demo_start_pose is not None:
        demo_start_mat = pose_to_mat(demo_start_pose)
        rel_start_mats = convert_pose_mat_rep(
            pose_mats,
            base_pose_mat=demo_start_mat,
            pose_rep="relative",
            backward=False,
        )
        rel_rot6d_wrt_start = np.array([rotmat_to_rot6d(p[:3, :3]) for p in rel_start_mats])
    else:
        rel_rot6d_wrt_start = np.zeros((len(pose_mats), 6), dtype=np.float32)

    obs_dict = {
        "camera0_rgb": images,
        "robot0_eef_pos": torch.from_numpy(rel_poses_6d[:, :3]).float().unsqueeze(0).to(device),
        "robot0_eef_rot_axis_angle": torch.from_numpy(rel_rot6d).float().unsqueeze(0).to(device),
        "robot0_eef_rot_axis_angle_wrt_start": torch.from_numpy(rel_rot6d_wrt_start).float().unsqueeze(0).to(device),
        "robot0_gripper_width": torch.from_numpy(obs["robot0_gripper_width"]).float().unsqueeze(0).to(device),
    }
    if "robot0_eef_wrench_left" in obs and obs["robot0_eef_wrench_left"] is not None:
        obs_dict["robot0_eef_wrench_left"] = torch.from_numpy(obs["robot0_eef_wrench_left"]).float().unsqueeze(0).to(device)
        obs_dict["robot0_eef_wrench_right"] = torch.from_numpy(obs["robot0_eef_wrench_right"]).float().unsqueeze(0).to(device)
    return obs_dict


def dataset_index_to_meta(dataset, global_idx):
    dset_id = dataset.item_id_to_dataset_id[int(global_idx)]
    local_idx = int(global_idx) - dataset.acc_dset_num[dset_id]
    dset = dataset._datasets[dset_id]
    return dset_id, local_idx, dset, dset.new_metas[local_idx]


def unnormalize_lingbot_action8(config, actions):
    q01 = torch.tensor(config.norm_stat["q01"], device=actions.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
    q99 = torch.tensor(config.norm_stat["q99"], device=actions.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
    unnorm = (actions.float() + 1.0) * 0.5 * (q99 - q01 + 1e-6) + q01
    return unnorm[:, config.used_action_channel_ids, :, :, :]


def dense_action8_at_steps(dense_action8, raw_steps, segment_start, action_per_frame=16):
    """Select raw zarr steps from [8, F, 16] LingBot dense action tensor."""
    selected = []
    valid = []
    total_frames = dense_action8.shape[1]
    for raw_step in raw_steps:
        rel_step = int(raw_step) - int(segment_start)
        frame_id = rel_step // action_per_frame + 1
        step_id = rel_step % action_per_frame
        ok = rel_step >= 0 and 0 <= frame_id < total_frames
        valid.append(ok)
        if ok:
            selected.append(dense_action8[:, frame_id, step_id])
        else:
            selected.append(np.zeros((dense_action8.shape[0],), dtype=np.float32))
    return np.stack(selected, axis=1).astype(np.float32), np.asarray(valid, dtype=bool)


def lingbot_dense_to_uva_trajectory(
    dense_action8,
    extractor,
    episode_idx,
    timesteps,
    future_offsets,
    segment_start=0,
):
    ep0_mat = episode_zero_pose_mat(extractor, episode_idx)
    out = []
    valid_all = []
    for timestep in timesteps:
        raw_steps = int(timestep) + np.asarray(future_offsets, dtype=np.int64)
        action8, valid = dense_action8_at_steps(
            dense_action8,
            raw_steps,
            segment_start=segment_start,
        )
        cur_mat = current_pose_mat(extractor, episode_idx, timestep)
        out.append(lingbot_actions_to_uva_pose10d(action8, ep0_mat, cur_mat))
        valid_all.append(valid)
    return np.asarray(out), np.asarray(valid_all)


def load_zarr_datapoints(zarr_path, episode_idx, timesteps):
    from extract_episodes_with_frames import ZarrEpisodeExtractor

    extractor = ZarrEpisodeExtractor(
        zarr_path=zarr_path,
        n_frames_history=4,
        n_frames_output=4,
        down_sample_steps=3,
        use_relative_pose=True,
        use_ft=True,
    )
    datapoints = []
    for timestep in timesteps:
        dp = extractor.load_datapoint(episode_idx, int(timestep))
        dp["episode_idx"] = episode_idx
        datapoints.append(dp)
    return extractor, datapoints


def run_lingbot_reconstruction(args):
    from configs import VA_CONFIGS
    from dataset import MultiLatentLeRobotDataset
    from modules.utils import load_transformer
    from reconstruct_validation_examples import (
        format_predictions,
        move_to_device,
        prepare_input_dict,
        reconstruct_clean,
    )
    from utils import FlowMatchScheduler

    config = VA_CONFIGS[args.lingbot_config_name]
    checkpoint_root = Path(args.lingbot_checkpoint_root or config.wan22_pretrained_model_name_or_path)
    device = torch.device(args.lingbot_device if torch.cuda.is_available() or args.lingbot_device == "cpu" else "cpu")
    dtype = config.param_dtype

    dataset = MultiLatentLeRobotDataset(config=config)
    _, val_indices, val_episode_info = dataset.get_episode_split_indices(
        validation_fraction=args.validation_fraction,
        validation_size=args.validation_size,
        validation_seed=args.validation_seed,
        return_episode_info=True,
    )
    if args.dataset_index is not None:
        sample_idx = int(args.dataset_index)
    else:
        sample_idx = int(val_indices[int(args.example_index)])

    dset_id, local_idx, _, meta = dataset_index_to_meta(dataset, sample_idx)
    episode_idx = int(meta["episode_index"])
    segment_start = int(meta["start_frame"])
    segment_end = int(meta["end_frame"])

    subset = Subset(dataset, [sample_idx])
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader))

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

    batch = move_to_device(batch, device)
    input_dict = prepare_input_dict(
        batch,
        config,
        latent_scheduler,
        action_scheduler,
        timestep_index,
        device,
    )
    with torch.no_grad():
        pred = transformer(input_dict, train_mode=True)
        _, action_pred = format_predictions(config, input_dict, pred)
        _, pred_clean_action = reconstruct_clean(
            input_dict,
            torch.zeros_like(input_dict["latent_dict"]["latent"]),
            action_pred,
            latent_scheduler,
            action_scheduler,
        )

    gt_action8 = unnormalize_lingbot_action8(config, input_dict["action_dict"]["latent"])[0, :, :, :, 0].detach().cpu().numpy()
    pred_action8 = unnormalize_lingbot_action8(config, pred_clean_action)[0, :, :, :, 0].detach().cpu().numpy()

    from extract_episodes_with_frames import ZarrEpisodeExtractor

    extractor = ZarrEpisodeExtractor(
        zarr_path=args.zarr_path,
        n_frames_history=4,
        n_frames_output=4,
        down_sample_steps=3,
        use_relative_pose=True,
        use_ft=True,
    )
    episode_len = int(extractor.episode_ends[episode_idx] - extractor.episode_starts[episode_idx])
    max_timestep = min(segment_end - 1 - max(args.future_offsets), episode_len - 1 - max(args.future_offsets))
    timesteps = np.arange(segment_start, max_timestep + 1, args.trajectory_stride, dtype=np.int64)
    if args.max_points and len(timesteps) > args.max_points:
        sample_ids = np.linspace(0, len(timesteps) - 1, args.max_points).round().astype(np.int64)
        timesteps = timesteps[sample_ids]

    lingbot_gt_uva, valid = lingbot_dense_to_uva_trajectory(
        gt_action8,
        extractor,
        episode_idx,
        timesteps,
        args.future_offsets,
        segment_start=segment_start,
    )
    lingbot_pred_uva, _ = lingbot_dense_to_uva_trajectory(
        pred_action8,
        extractor,
        episode_idx,
        timesteps,
        args.future_offsets,
        segment_start=segment_start,
    )

    zarr_gt = []
    for timestep in timesteps:
        dp = extractor.load_datapoint(episode_idx, int(timestep))
        zarr_gt.append(dp["actions_relative"])
    zarr_gt = np.asarray(zarr_gt)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "lingbot_reconstruction_uva.npz",
        timesteps=timesteps,
        episode_idx=np.asarray(episode_idx),
        dataset_index=np.asarray(sample_idx),
        dataset_id=np.asarray(dset_id),
        local_index=np.asarray(local_idx),
        segment_start=np.asarray(segment_start),
        segment_end=np.asarray(segment_end),
        future_offsets=np.asarray(args.future_offsets),
        zarr_gt_uva_pose10d=zarr_gt,
        lingbot_gt_uva_pose10d=lingbot_gt_uva,
        lingbot_recon_pred_uva_pose10d=lingbot_pred_uva,
        valid_future_mask=valid,
        dense_gt_action8=gt_action8,
        dense_pred_action8=pred_action8,
        val_episode_info=np.asarray(str(val_episode_info)),
    )
    gt_diff = np.linalg.norm(zarr_gt[..., :3] - lingbot_gt_uva[..., :3], axis=-1)
    pred_diff = np.linalg.norm(zarr_gt[..., :3] - lingbot_pred_uva[..., :3], axis=-1)
    print(f"Saved LingBot reconstruction UVA data to: {(output_dir / 'lingbot_reconstruction_uva.npz').resolve()}")
    print(f"episode={episode_idx} dataset_index={sample_idx} segment={segment_start}:{segment_end} points={len(timesteps)}")
    print(f"zarr GT vs LingBot-dense GT xyz L2 mean/max: {gt_diff.mean():.8f}/{gt_diff.max():.8f}")
    print(f"zarr GT vs LingBot recon pred xyz L2 mean/max: {pred_diff.mean():.8f}/{pred_diff.max():.8f}")


def run_uva(args):
    from model_utils import load_model as load_uva_model
    from model_utils import predict as predict_uva

    lingbot_path = Path(args.output_dir) / "lingbot_reconstruction_uva.npz"
    if not lingbot_path.exists():
        raise FileNotFoundError(f"Run --mode lingbot first so {lingbot_path} exists.")
    ling = np.load(lingbot_path)
    timesteps = ling["timesteps"]
    episode_idx = int(ling["episode_idx"])
    _, datapoints = load_zarr_datapoints(args.zarr_path, episode_idx, timesteps)

    policy, _, cfg = load_uva_model(args.uva_checkpoint, args.uva_device, args.ft_normalizer_checkpoint)
    preds = []
    for dp in tqdm(datapoints, desc="UVA"):
        obs_dict = build_uva_obs(dp, args.uva_device)
        result = predict_uva(
            policy,
            obs_dict,
            cfg,
            task_name=args.task_name,
            estimate_ft=args.estimate_ft,
            device=args.uva_device,
        )
        preds.append(result["action"].detach().cpu().numpy()[0])
    del policy
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    np.savez_compressed(
        Path(args.output_dir) / "uva_reconstruction_trajectory.npz",
        timesteps=timesteps,
        episode_idx=np.asarray(episode_idx),
        uva_pred_pose10d=np.asarray(preds),
    )
    print(f"Saved UVA trajectory predictions to: {(Path(args.output_dir) / 'uva_reconstruction_trajectory.npz').resolve()}")


def plot_timeline(args):
    output_dir = Path(args.output_dir)
    ling = np.load(output_dir / "lingbot_reconstruction_uva.npz")
    uva_path = output_dir / "uva_reconstruction_trajectory.npz"
    uva = np.load(uva_path) if uva_path.exists() else None

    timesteps = ling["timesteps"]
    zarr_gt = ling["zarr_gt_uva_pose10d"]
    ling_gt = ling["lingbot_gt_uva_pose10d"]
    ling_pred = ling["lingbot_recon_pred_uva_pose10d"]
    uva_pred = uva["uva_pred_pose10d"][:, : len(ling["future_offsets"])] if uva is not None else None

    def aa(arr):
        out = np.asarray([[pose10d_to_axis_angle(x) for x in chunk] for chunk in arr])
        out[..., 3:6] = np.degrees(out[..., 3:6])
        return out

    gt_aa = aa(zarr_gt)
    ling_gt_aa = aa(ling_gt)
    ling_pred_aa = aa(ling_pred)
    uva_aa = aa(uva_pred) if uva_pred is not None else None

    labels = ["x", "y", "z", "rx_deg", "ry_deg", "rz_deg", "gripper"]
    fig, axes = plt.subplots(4, 2, figsize=(17, 13), sharex=True)
    axes = axes.flatten()
    for dim, label in enumerate(labels):
        ax = axes[dim]
        ax.plot(timesteps, gt_aa[:, 0, dim], "k-", linewidth=2.0, label="zarr GT")
        ax.plot(timesteps, ling_gt_aa[:, 0, dim], color="0.55", linestyle=":", linewidth=1.5, label="LingBot dense GT")
        ax.plot(timesteps, ling_pred_aa[:, 0, dim], "C3--", linewidth=1.5, label="LingBot recon")
        if uva_aa is not None:
            ax.plot(timesteps, uva_aa[:, 0, dim], "C0--", linewidth=1.5, label="UVA")
        ax.set_title(f"First future action (+{int(ling['future_offsets'][0])}): {label}")
        ax.grid(alpha=0.25)
        if dim == 0:
            ax.legend(fontsize=8)
    axes[-1].axis("off")
    axes[-2].set_xlabel("Episode timestep")
    fig.tight_layout()
    fig.savefig(output_dir / "reconstruction_uva_first_action_timeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    for horizon in range(len(ling["future_offsets"])):
        fig, axes = plt.subplots(4, 2, figsize=(17, 13), sharex=True)
        axes = axes.flatten()
        for dim, label in enumerate(labels):
            ax = axes[dim]
            ax.plot(timesteps, gt_aa[:, horizon, dim], "k-", linewidth=2.0, label="zarr GT")
            ax.plot(timesteps, ling_gt_aa[:, horizon, dim], color="0.55", linestyle=":", linewidth=1.5, label="LingBot dense GT")
            ax.plot(timesteps, ling_pred_aa[:, horizon, dim], "C3--", linewidth=1.5, label="LingBot recon")
            if uva_aa is not None:
                ax.plot(timesteps, uva_aa[:, horizon, dim], "C0--", linewidth=1.5, label="UVA")
            ax.set_title(f"+{int(ling['future_offsets'][horizon])}: {label}")
            ax.grid(alpha=0.25)
            if dim == 0:
                ax.legend(fontsize=8)
        axes[-1].axis("off")
        axes[-2].set_xlabel("Episode timestep")
        fig.tight_layout()
        fig.savefig(output_dir / f"reconstruction_uva_horizon_{horizon:02d}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    gt_diff = np.linalg.norm(zarr_gt[..., :3] - ling_gt[..., :3], axis=-1)
    ling_pred_diff = np.linalg.norm(zarr_gt[..., :3] - ling_pred[..., :3], axis=-1)
    print(f"Saved plots to: {output_dir.resolve()}")
    print(f"zarr GT vs LingBot dense GT xyz L2 mean/max: {gt_diff.mean():.8f}/{gt_diff.max():.8f}")
    print(f"zarr GT vs LingBot recon xyz L2 mean/max: {ling_pred_diff.mean():.8f}/{ling_pred_diff.max():.8f}")
    if uva_pred is not None:
        uva_diff = np.linalg.norm(zarr_gt[..., :3] - uva_pred[..., :3], axis=-1)
        print(f"zarr GT vs UVA xyz L2 mean/max: {uva_diff.mean():.8f}/{uva_diff.max():.8f}")


def subprocess_args(args, mode):
    skip = {"mode", "uva_python", "lingbot_python"}
    argv = [str(Path(__file__).resolve()), "--mode", mode]
    for key, value in vars(args).items():
        if key in skip:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        elif isinstance(value, list):
            argv.append(flag)
            argv.extend(str(v) for v in value)
        elif value is not None:
            argv.extend([flag, str(value)])
    return argv


def run_all(args):
    commands = [
        [args.lingbot_python] + subprocess_args(args, "lingbot"),
        [args.uva_python] + subprocess_args(args, "uva"),
        [args.lingbot_python] + subprocess_args(args, "plot"),
    ]
    for cmd in commands:
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Reconstruction-aligned LingBot/UVA action comparison.")
    parser.add_argument("--mode", choices=["all", "lingbot", "uva", "plot"], default="all")
    parser.add_argument("--uva-python", default="/local/real/jvclark/miniconda3/envs/uva_1/bin/python")
    parser.add_argument("--lingbot-python", default="/local/real/jvclark/miniconda3/envs/lingbot-va/bin/python")
    parser.add_argument("--zarr-path", default="/local/real/jvclark/unified_video_action/uva/umi_data/umi_ft/WBW2-addition-wristOvershoot-harderGrasp-ft36_true2.zarr")
    parser.add_argument("--task-name", default="erase the drawing")
    parser.add_argument("--output-dir", default="/store/real/jvclark/lingbot-va/action_compare_out/reconstruction_step_8000")
    parser.add_argument("--uva-checkpoint", default="/store/real/jvclark/unified_video_action/checkpoints/mycheckpoints/uva_umi_multitask_video_action_vaseft_wbwnoft/checkpoints/latest.ckpt")
    parser.add_argument("--uva-device", default="cuda:0")
    parser.add_argument("--ft-normalizer-checkpoint", default=None)
    parser.add_argument("--estimate-ft", action="store_true")
    parser.add_argument("--lingbot-config-name", default="umi_wbw_train")
    parser.add_argument("--lingbot-checkpoint-root", default="/store/real/jvclark/lingbot-va/train_out/checkpoints/checkpoint_step_8000")
    parser.add_argument("--lingbot-device", default="cuda:0")
    parser.add_argument("--example-index", type=int, default=0)
    parser.add_argument("--dataset-index", type=int, default=None)
    parser.add_argument("--validation-size", type=int, default=128)
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--validation-seed", type=int, default=42)
    parser.add_argument("--noise-timestep-index", type=int, default=500)
    parser.add_argument("--trajectory-stride", type=int, default=12)
    parser.add_argument("--max-points", type=int, default=60)
    parser.add_argument("--future-offsets", type=int, nargs=4, default=[3, 6, 9, 12])
    return parser.parse_args()


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.mode == "all":
        run_all(args)
    elif args.mode == "lingbot":
        run_lingbot_reconstruction(args)
    elif args.mode == "uva":
        run_uva(args)
    elif args.mode == "plot":
        plot_timeline(args)


if __name__ == "__main__":
    main()
