#!/usr/bin/env python3
"""Compare UVA and LingBot action chunks on the same UMI zarr examples.

The script samples datapoints from a zarr episode, runs UVA and LingBot on the
same observation windows, converts LingBot's episode-frame-0 relative actions
back into UVA's current-frame relative pose10d format, and writes plots.
"""

import argparse
import gc
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from tqdm import tqdm


LINGBOT_ROOT = Path(__file__).resolve().parent
UVA_ROOT = Path("/store/real/jvclark/unified_video_action")
sys.path.insert(0, str(UVA_ROOT))
sys.path.insert(0, str(LINGBOT_ROOT / "wan_va"))

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


def build_uva_obs(dp, device):
    obs = dp["observation"]
    images = torch.from_numpy(obs["camera0_rgb"]).float().permute(0, 3, 1, 2) / 255.0
    images = images.unsqueeze(0).to(device)

    poses_6d = np.concatenate(
        [obs["robot0_eef_pos"], obs["robot0_eef_rot_axis_angle"]],
        axis=-1,
    )
    pose_mats = pose_to_mat(poses_6d)
    current_pose_mat = pose_mats[-1]
    rel_pose_mats = convert_pose_mat_rep(
        pose_mats,
        base_pose_mat=current_pose_mat,
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


def run_uva_predictions(datapoints, checkpoint, device, task_name, estimate_ft=False, ft_normalizer_checkpoint=None):
    from model_utils import load_model as load_uva_model
    from model_utils import predict as predict_uva

    policy, _, cfg = load_uva_model(checkpoint, device, ft_normalizer_checkpoint)
    preds = []
    for dp in tqdm(datapoints, desc="UVA"):
        obs_dict = build_uva_obs(dp, device)
        result = predict_uva(
            policy,
            obs_dict,
            cfg,
            task_name=task_name,
            estimate_ft=estimate_ft,
            device=device,
        )
        preds.append(result["action"].detach().cpu().numpy()[0])
    del policy
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.asarray(preds)


class LingBotActionRunner:
    """Small single-process LingBot runner for action chunks.

    It mirrors wan_va_server.py, but lets the transformer come from a finetuned
    checkpoint while VAE/tokenizer/text_encoder come from the base model root.
    """

    def __init__(self, config_name, base_root, checkpoint_root, device="cuda:0", save_root="/tmp/lingbot_action_compare"):
        from configs import VA_CONFIGS
        from modules.utils import WanVAEStreamingWrapper, load_text_encoder, load_tokenizer, load_transformer, load_vae
        from utils import FlowMatchScheduler

        self.WanVAEStreamingWrapper = WanVAEStreamingWrapper
        self.load_text_encoder = load_text_encoder
        self.load_tokenizer = load_tokenizer
        self.load_transformer = load_transformer
        self.load_vae = load_vae
        self.FlowMatchScheduler = FlowMatchScheduler

        self.config = VA_CONFIGS[config_name]
        self.base_root = Path(base_root)
        self.checkpoint_root = Path(checkpoint_root)
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.dtype = self.config.param_dtype
        self.cache_name = "pos"
        self.save_root = save_root

        self.scheduler = FlowMatchScheduler(shift=self.config.snr_shift, sigma_min=0.0, extra_one_step=True)
        self.action_scheduler = FlowMatchScheduler(shift=self.config.action_snr_shift, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        self.action_scheduler.set_timesteps(1000, training=True)

        self.vae = load_vae(str(self.base_root / "vae"), torch_dtype=self.dtype, torch_device=self.device)
        self.streaming_vae = WanVAEStreamingWrapper(self.vae)
        self.tokenizer = load_tokenizer(str(self.base_root / "tokenizer"))
        self.text_encoder = load_text_encoder(str(self.base_root / "text_encoder"), torch_dtype=self.dtype, torch_device="cpu")
        self.transformer = load_transformer(
            str(self.checkpoint_root / "transformer"),
            torch_dtype=self.dtype,
            torch_device=self.device,
        )
        self.transformer.eval().requires_grad_(False)

        self.env_type = self.config.env_type
        self.height = self.config.height
        self.width = self.config.width
        self.action_per_frame = self.config.action_per_frame
        self.latent_height = self.height // 16
        self.latent_width = self.width // 16 * len(self.config.obs_cam_keys)
        self.use_cfg = (self.config.guidance_scale > 1) or (self.config.action_guidance_scale > 1)

        self.action_mask = torch.zeros([self.config.action_dim], dtype=torch.bool)
        self.action_mask[self.config.used_action_channel_ids] = True
        self.actions_q01 = torch.tensor(self.config.norm_stat["q01"], dtype=torch.float32).reshape(-1, 1, 1)
        self.actions_q99 = torch.tensor(self.config.norm_stat["q99"], dtype=torch.float32).reshape(-1, 1, 1)

    def encode_prompt(self, prompt, max_sequence_length=512):
        from diffusers.pipelines.wan.pipeline_wan import prompt_clean

        prompt = [prompt_clean(prompt)]
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        seq_lens = text_inputs.attention_mask.gt(0).sum(dim=1).long()
        text_encoder_device = next(self.text_encoder.parameters()).device
        prompt_embeds = self.text_encoder(
            text_inputs.input_ids.to(text_encoder_device),
            text_inputs.attention_mask.to(text_encoder_device),
        ).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=self.device)
        prompt_embeds = torch.stack(
            [
                torch.cat([u[:v], u.new_zeros(max_sequence_length - v, u.size(1))])
                for u, v in zip(prompt_embeds, seq_lens)
            ],
            dim=0,
        )
        negative_embeds = self.encode_negative_prompt(max_sequence_length)
        return prompt_embeds, negative_embeds

    def encode_negative_prompt(self, max_sequence_length=512):
        text_inputs = self.tokenizer(
            [""],
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        seq_lens = text_inputs.attention_mask.gt(0).sum(dim=1).long()
        text_encoder_device = next(self.text_encoder.parameters()).device
        embeds = self.text_encoder(
            text_inputs.input_ids.to(text_encoder_device),
            text_inputs.attention_mask.to(text_encoder_device),
        ).last_hidden_state
        embeds = embeds.to(dtype=self.dtype, device=self.device)
        return torch.stack(
            [torch.cat([u[:v], u.new_zeros(max_sequence_length - v, u.size(1))]) for u, v in zip(embeds, seq_lens)],
            dim=0,
        )

    def reset(self, prompt):
        self.transformer.clear_cache(self.cache_name)
        self.streaming_vae.clear_cache()
        patch_size = self.config.patch_size
        latent_token_per_chunk = (
            self.config.frame_chunk_size * self.latent_height * self.latent_width
        ) // (patch_size[0] * patch_size[1] * patch_size[2])
        action_token_per_chunk = self.config.frame_chunk_size * self.action_per_frame
        self.transformer.create_empty_cache(
            self.cache_name,
            self.config.attn_window,
            latent_token_per_chunk,
            action_token_per_chunk,
            dtype=self.dtype,
            device=self.device,
            batch_size=2 if self.use_cfg else 1,
        )
        self.prompt_embeds, self.negative_prompt_embeds = self.encode_prompt(prompt)

    def normalize_latents(self, latents, latents_mean, latents_std):
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(device=latents.device)
        return ((latents.float() - latents_mean) * latents_std).to(latents)

    def encode_obs(self, obs_list):
        videos = []
        for key in self.config.obs_cam_keys:
            history = torch.from_numpy(np.stack([frame[key] for frame in obs_list])).float().permute(3, 0, 1, 2)
            history = F.interpolate(
                history,
                size=(self.height, self.width),
                mode="bilinear",
                align_corners=False,
            ).unsqueeze(0)
            videos.append(history)
        videos = torch.cat(videos, dim=0) / 255.0 * 2.0 - 1.0
        vae_device = next(self.vae.parameters()).device
        videos = videos.to(vae_device).to(self.dtype)
        # Full encode handles Wan's causal temporal grouping correctly for a
        # fresh multi-frame history window. The streaming wrapper is for
        # sequential cache updates and can shape-mismatch on the first call.
        mu = self.vae.encode(videos).latent_dist.mean
        latents_mean = torch.tensor(self.vae.config.latents_mean).to(mu.device)
        latents_std = torch.tensor(self.vae.config.latents_std).to(mu.device)
        mu_norm = self.normalize_latents(mu, latents_mean, 1.0 / latents_std)
        video_latent = torch.cat(mu_norm.split(1, dim=0), dim=-1)
        return video_latent.to(self.device)

    def repeat_input_for_cfg(self, input_dict):
        if self.use_cfg:
            input_dict["noisy_latents"] = input_dict["noisy_latents"].repeat(2, 1, 1, 1, 1)
            input_dict["text_emb"] = torch.cat(
                [self.prompt_embeds.to(self.dtype).clone(), self.negative_prompt_embeds.to(self.dtype).clone()],
                dim=0,
            )
            input_dict["grid_id"] = input_dict["grid_id"][None].repeat(2, 1, 1)
            input_dict["timesteps"] = input_dict["timesteps"][None].repeat(2, 1)
        else:
            input_dict["grid_id"] = input_dict["grid_id"][None]
            input_dict["timesteps"] = input_dict["timesteps"][None]
        return input_dict

    def prepare_latent_input(self, latent_model_input=None, action_model_input=None, latent_t=0, action_t=0, latent_cond=None, action_cond=None, frame_st_id=0):
        from utils import get_mesh_id

        out = {}
        if latent_model_input is not None:
            out["latent_res_lst"] = {
                "noisy_latents": latent_model_input,
                "timesteps": torch.ones([latent_model_input.shape[2]], dtype=torch.float32, device=self.device) * latent_t,
                "grid_id": get_mesh_id(
                    latent_model_input.shape[-3] // self.config.patch_size[0],
                    latent_model_input.shape[-2] // self.config.patch_size[1],
                    latent_model_input.shape[-1] // self.config.patch_size[2],
                    0,
                    1,
                    frame_st_id,
                ).to(self.device),
                "text_emb": self.prompt_embeds.to(self.dtype).clone(),
            }
            if latent_cond is not None:
                out["latent_res_lst"]["noisy_latents"][:, :, 0:1] = latent_cond[:, :, 0:1]
                out["latent_res_lst"]["timesteps"][0:1] *= 0
        if action_model_input is not None:
            out["action_res_lst"] = {
                "noisy_latents": action_model_input,
                "timesteps": torch.ones([action_model_input.shape[2]], dtype=torch.float32, device=self.device) * action_t,
                "grid_id": get_mesh_id(
                    action_model_input.shape[-3],
                    action_model_input.shape[-2],
                    action_model_input.shape[-1],
                    1,
                    1,
                    frame_st_id,
                    action=True,
                ).to(self.device),
                "text_emb": self.prompt_embeds.to(self.dtype).clone(),
            }
            if action_cond is not None:
                out["action_res_lst"]["noisy_latents"][:, :, 0:1] = action_cond[:, :, 0:1]
                out["action_res_lst"]["timesteps"][0:1] *= 0
            out["action_res_lst"]["noisy_latents"][:, ~self.action_mask] *= 0
        return out

    def preprocess_action(self, action):
        action_model_input = torch.from_numpy(action).float()
        action_model_input_padded = F.pad(
            action_model_input,
            [0, 0, 0, 0, 0, 1],
            mode="constant",
            value=0,
        )
        action_model_input = action_model_input_padded[self.config.inverse_used_action_channel_ids]
        action_model_input = (action_model_input - self.actions_q01) / (
            self.actions_q99 - self.actions_q01 + 1e-6
        ) * 2.0 - 1.0
        return action_model_input.unsqueeze(0).unsqueeze(-1).to(self.device, self.dtype)

    def postprocess_action(self, action):
        action = action.cpu()[0, ..., 0]
        action = (action + 1) / 2 * (self.actions_q99 - self.actions_q01 + 1e-6) + self.actions_q01
        action = action.detach().cpu().numpy()
        return action[self.config.used_action_channel_ids]

    @torch.no_grad()
    def compute_history_cache(self, obs_list, action_history, prompt):
        self.reset(prompt)
        latent_model_input = self.encode_obs(obs_list)
        action_model_input = self.preprocess_action(action_history).to(latent_model_input)
        input_dict = self.prepare_latent_input(
            latent_model_input=latent_model_input,
            action_model_input=action_model_input,
            frame_st_id=0,
        )
        self.transformer(
            self.repeat_input_for_cfg(input_dict["latent_res_lst"]),
            update_cache=2,
            cache_name=self.cache_name,
            action_mode=False,
        )
        self.transformer(
            self.repeat_input_for_cfg(input_dict["action_res_lst"]),
            update_cache=2,
            cache_name=self.cache_name,
            action_mode=True,
        )
        return int(latent_model_input.shape[2])

    @torch.no_grad()
    def sample_next_chunk(self, frame_st_id):
        from einops import rearrange
        from utils import data_seq_to_patch

        frame_chunk_size = self.config.frame_chunk_size
        latents = torch.randn(
            1,
            48,
            frame_chunk_size,
            self.latent_height,
            self.latent_width,
            device=self.device,
            dtype=self.dtype,
        )
        actions = torch.randn(
            1,
            self.config.action_dim,
            frame_chunk_size,
            self.action_per_frame,
            1,
            device=self.device,
            dtype=self.dtype,
        )
        self.scheduler.set_timesteps(self.config.num_inference_steps)
        self.action_scheduler.set_timesteps(self.config.action_num_inference_steps)
        timesteps = F.pad(self.scheduler.timesteps, (0, 1), mode="constant", value=0)
        if self.config.video_exec_step != -1:
            timesteps = timesteps[: self.config.video_exec_step]
        action_timesteps = F.pad(self.action_scheduler.timesteps, (0, 1), mode="constant", value=0)

        for i, t in enumerate(timesteps):
            last_step = i == len(timesteps) - 1
            input_dict = self.prepare_latent_input(
                latents,
                None,
                t,
                t,
                None,
                None,
                frame_st_id=frame_st_id,
            )
            pred = self.transformer(
                self.repeat_input_for_cfg(input_dict["latent_res_lst"]),
                update_cache=1 if last_step else 0,
                cache_name=self.cache_name,
                action_mode=False,
            )
            if not last_step or self.config.video_exec_step != -1:
                pred = data_seq_to_patch(
                    self.config.patch_size,
                    pred,
                    frame_chunk_size,
                    self.latent_height,
                    self.latent_width,
                    batch_size=2 if self.use_cfg else 1,
                )
                if self.config.guidance_scale > 1:
                    pred = pred[1:] + self.config.guidance_scale * (pred[:1] - pred[1:])
                else:
                    pred = pred[:1]
                latents = self.scheduler.step(pred, t, latents, return_dict=False)

        for i, t in enumerate(action_timesteps):
            last_step = i == len(action_timesteps) - 1
            input_dict = self.prepare_latent_input(
                None,
                actions,
                t,
                t,
                None,
                None,
                frame_st_id=frame_st_id,
            )
            pred = self.transformer(
                self.repeat_input_for_cfg(input_dict["action_res_lst"]),
                update_cache=1 if last_step else 0,
                cache_name=self.cache_name,
                action_mode=True,
            )
            if not last_step:
                pred = rearrange(pred, "b (f n) c -> b c f n 1", f=frame_chunk_size)
                if self.config.action_guidance_scale > 1:
                    pred = pred[1:] + self.config.action_guidance_scale * (pred[:1] - pred[1:])
                else:
                    pred = pred[:1]
                actions = self.action_scheduler.step(pred, t, actions, return_dict=False)

        actions[:, ~self.action_mask] *= 0
        return self.postprocess_action(actions)

    @torch.no_grad()
    def predict_after_history(self, obs_list, action_history, prompt):
        frame_st_id = self.compute_history_cache(obs_list, action_history, prompt)
        return self.sample_next_chunk(frame_st_id)


def lingbot_obs_from_dp(dp, camera_key):
    return [{camera_key: frame} for frame in dp["observation"]["camera0_rgb"]]


def lingbot_obs_history_from_zarr(extractor, episode_idx, timestep, camera_key, stride=4, max_frames=0):
    start = int(extractor.episode_starts[episode_idx])
    end = int(extractor.episode_ends[episode_idx])
    ep_len = end - start
    local_indices = list(range(0, min(timestep, ep_len - 1) + 1, stride))
    if len(local_indices) == 0 or local_indices[-1] != min(timestep, ep_len - 1):
        local_indices.append(min(timestep, ep_len - 1))
    if max_frames and len(local_indices) > max_frames:
        local_indices = local_indices[-max_frames:]
    frames = np.asarray([extractor.data_store["camera0_rgb"][start + idx] for idx in local_indices])
    return [{camera_key: frame} for frame in frames], local_indices


def relative_action_episode0(pos, rot_axis_angle, grip):
    rot = Rotation.from_rotvec(rot_axis_angle.astype(np.float64))
    rel_xyz = (pos - pos[0:1]).astype(np.float32)
    rel_rot = Rotation.from_rotvec(rot_axis_angle[0:1].astype(np.float64)).inv() * rot
    rel_quat_xyzw = rel_rot.as_quat().astype(np.float32)
    return np.concatenate([rel_xyz, rel_quat_xyzw, grip.astype(np.float32)], axis=1)


def lingbot_action_history_from_zarr(extractor, episode_idx, first_obs_timestep, latent_frames):
    start = int(extractor.episode_starts[episode_idx])
    end = int(extractor.episode_ends[episode_idx])
    ep_len = end - start
    action_per_frame = 16
    required_steps = int(latent_frames) * action_per_frame
    packed = np.zeros((required_steps, 8), dtype=np.float32)
    # Training pads one whole action frame with zeros before the segment
    # actions, so the first latent frame has a zero action condition.
    history_steps = max(required_steps - action_per_frame, 0)
    step_indices = np.arange(int(first_obs_timestep), int(first_obs_timestep) + history_steps)
    valid = (step_indices >= 0) & (step_indices < ep_len)
    if history_steps > 0 and valid.any():
        valid_indices = step_indices[valid].astype(np.int64)
        pos = np.asarray([extractor.data_store["robot0_eef_pos"][start + idx] for idx in valid_indices])
        rot = np.asarray([extractor.data_store["robot0_eef_rot_axis_angle"][start + idx] for idx in valid_indices])
        grip = np.asarray([extractor.data_store["robot0_gripper_width"][start + idx] for idx in valid_indices])
        valid_positions = np.arange(action_per_frame, required_steps)[valid]
        packed[valid_positions] = relative_action_episode0(pos, rot, grip).astype(np.float32)
    return rearrange_np(packed, latent_frames, action_per_frame)


def rearrange_np(action_steps, latent_frames, action_per_frame):
    return action_steps.reshape(latent_frames, action_per_frame, action_steps.shape[-1]).transpose(2, 0, 1)


def episode_zero_pose_mat(extractor, episode_idx):
    start = extractor.episode_starts[episode_idx]
    pos = np.asarray(extractor.data_store["robot0_eef_pos"][start])
    rot = np.asarray(extractor.data_store["robot0_eef_rot_axis_angle"][start])
    return pose_to_mat(np.concatenate([pos, rot]))


def current_pose_mat(extractor, episode_idx, timestep):
    start = extractor.episode_starts[episode_idx]
    idx = start + timestep
    pos = np.asarray(extractor.data_store["robot0_eef_pos"][idx])
    rot = np.asarray(extractor.data_store["robot0_eef_rot_axis_angle"][idx])
    return pose_to_mat(np.concatenate([pos, rot]))


def lingbot_actions_to_uva_pose10d(action8_frame, ep0_pose_mat, cur_pose_mat):
    # action8_frame: [8, N], LingBot UMI action format:
    #   xyz is world-frame delta from episode frame 0, not SE(3)-relative xyz.
    #   quat_xyzw is rotation relative to episode frame 0.
    mats = []
    grippers = []
    ep0_pos = ep0_pose_mat[:3, 3]
    ep0_rot = Rotation.from_matrix(ep0_pose_mat[:3, :3])
    for k in range(action8_frame.shape[1]):
        abs_pose = np.eye(4, dtype=np.float64)
        abs_pose[:3, 3] = ep0_pos + action8_frame[:3, k]
        abs_pose[:3, :3] = (ep0_rot * Rotation.from_quat(action8_frame[3:7, k])).as_matrix()
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


def flatten_lingbot_action_chunk(action8):
    # LingBot returns [C, frame_chunk_size, action_per_frame]. Flatten the two
    # time axes into a chronological [C, T] chunk.
    return action8.reshape(action8.shape[0], -1)


def run_lingbot_predictions(datapoints, extractor, episode_idx, args):
    runner = LingBotActionRunner(
        config_name=args.lingbot_config_name,
        base_root=args.lingbot_base_root,
        checkpoint_root=args.lingbot_checkpoint_root,
        device=args.lingbot_device,
        save_root=args.output_dir,
    )
    ep0_mat = episode_zero_pose_mat(extractor, episode_idx)
    preds_4 = []
    raw_chunks = []
    sample_ids_all = []
    pred_starts = []
    for dp in tqdm(datapoints, desc="LingBot"):
        obs_list, obs_local_indices = lingbot_obs_history_from_zarr(
            extractor,
            episode_idx,
            dp["timestep"],
            runner.config.obs_cam_keys[0],
            stride=args.lingbot_history_stride,
            max_frames=args.lingbot_max_history_frames,
        )
        latent_frames = (len(obs_list) - 1) // 4 + 1
        action_history = lingbot_action_history_from_zarr(
            extractor,
            episode_idx,
            obs_local_indices[0],
            latent_frames,
        )
        action8 = runner.predict_after_history(obs_list, action_history, args.task_name)
        flat_action8 = flatten_lingbot_action_chunk(action8)
        cur_mat = current_pose_mat(extractor, episode_idx, dp["timestep"])
        chunk_uva = lingbot_actions_to_uva_pose10d(flat_action8, ep0_mat, cur_mat)
        pred_start = int(obs_local_indices[0]) + (latent_frames - 1) * runner.action_per_frame
        if args.lingbot_sample_indices is not None:
            sample_ids = np.asarray(args.lingbot_sample_indices, dtype=np.int64)
        else:
            target_timesteps = int(dp["timestep"]) + np.asarray(args.future_offsets, dtype=np.int64)
            sample_ids = target_timesteps - pred_start
        clipped_sample_ids = np.clip(sample_ids, 0, len(chunk_uva) - 1).astype(np.int64)
        preds_4.append(chunk_uva[clipped_sample_ids])
        raw_chunks.append(chunk_uva)
        sample_ids_all.append(clipped_sample_ids)
        pred_starts.append(pred_start)
    del runner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.asarray(preds_4), np.asarray(raw_chunks), np.asarray(sample_ids_all), np.asarray(pred_starts)


def plot_episode_timeline(timesteps, gt, uva, lingbot, output_path):
    gt_aa = np.asarray([[pose10d_to_axis_angle(a) for a in chunk] for chunk in gt])
    uva_aa = np.asarray([[pose10d_to_axis_angle(a) for a in chunk] for chunk in uva])
    ling_aa = np.asarray([[pose10d_to_axis_angle(a) for a in chunk] for chunk in lingbot])

    labels = ["x", "y", "z", "rx_deg", "ry_deg", "rz_deg", "gripper"]
    gt_first = gt_aa[:, 0].copy()
    uva_first = uva_aa[:, 0].copy()
    ling_first = ling_aa[:, 0].copy()
    gt_first[:, 3:6] = np.degrees(gt_first[:, 3:6])
    uva_first[:, 3:6] = np.degrees(uva_first[:, 3:6])
    ling_first[:, 3:6] = np.degrees(ling_first[:, 3:6])

    fig, axes = plt.subplots(4, 2, figsize=(16, 13), sharex=True)
    axes = axes.flatten()
    for i, label in enumerate(labels):
        ax = axes[i]
        ax.plot(timesteps, gt_first[:, i], "k-", label="GT", linewidth=2)
        ax.plot(timesteps, uva_first[:, i], "C0--", label="UVA", linewidth=1.6)
        ax.plot(timesteps, ling_first[:, i], "C3--", label="LingBot", linewidth=1.6)
        ax.set_title(f"First future action: {label}")
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend()
    axes[-1].axis("off")
    axes[-2].set_xlabel("Episode timestep")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_chunk_panel(dp, gt, uva, lingbot, lingbot_full, output_path, lingbot_sample_ids=None, future_offsets=None):
    gt_aa = np.asarray([pose10d_to_axis_angle(a) for a in gt])
    uva_aa = np.asarray([pose10d_to_axis_angle(a) for a in uva])
    ling_aa = np.asarray([pose10d_to_axis_angle(a) for a in lingbot])
    ling_full_aa = np.asarray([pose10d_to_axis_angle(a) for a in lingbot_full])
    gt_aa[:, 3:6] = np.degrees(gt_aa[:, 3:6])
    uva_aa[:, 3:6] = np.degrees(uva_aa[:, 3:6])
    ling_aa[:, 3:6] = np.degrees(ling_aa[:, 3:6])
    ling_full_aa[:, 3:6] = np.degrees(ling_full_aa[:, 3:6])

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 4, hspace=0.45, wspace=0.3)
    images = dp["observation"]["camera0_rgb"]
    for i in range(4):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(images[i])
        ax.set_title(f"history {i}")
        ax.axis("off")

    dims = [
        ("x", 0),
        ("y", 1),
        ("z", 2),
        ("rx deg", 3),
        ("ry deg", 4),
        ("rz deg", 5),
        ("gripper", 6),
    ]
    x4 = np.arange(4)
    xfull = np.arange(len(ling_full_aa))
    for plot_i, (name, dim) in enumerate(dims):
        row = 1 + plot_i // 3
        col = plot_i % 3
        ax = fig.add_subplot(gs[row, col])
        ax.plot(x4, gt_aa[:, dim], "ko-", label="GT", linewidth=2)
        ax.plot(x4, uva_aa[:4, dim], "C0o--", label="UVA", linewidth=1.5)
        ax.plot(x4, ling_aa[:, dim], "C3o--", label="LingBot sampled", linewidth=1.5)
        ax.plot(xfull, ling_full_aa[:, dim], color="C3", alpha=0.22, linewidth=1.0, label=f"LingBot full {len(ling_full_aa)}")
        ax.set_title(name)
        ax.grid(alpha=0.25)
        if plot_i == 0:
            ax.legend(fontsize=8)

    ax_text = fig.add_subplot(gs[3, 3])
    ax_text.axis("off")
    ax_text.text(
        0,
        1,
        f"Episode {dp['episode_idx']}\n"
        f"Timestep {dp['timestep']}\n"
        f"Future offsets {list(future_offsets) if future_offsets is not None else '[3, 6, 9, 12]'}\n"
        f"LingBot indices {list(lingbot_sample_ids) if lingbot_sample_ids is not None else 'unknown'}\n\n"
        "All actions shown in UVA format:\n"
        "relative pose10d w.r.t. current frame.\n\n"
        "LingBot full curve is its generated\n"
        "raw-timestep action chunk; sampled\n"
        "points are matched by episode time.",
        va="top",
        family="monospace",
    )
    fig.suptitle("Action Chunk Comparison", fontsize=16, fontweight="bold")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare UVA and LingBot actions on the same zarr datapoints.")
    parser.add_argument("--mode", choices=["all", "uva", "lingbot", "plot"], default="all")
    parser.add_argument("--uva-python", default="/local/real/jvclark/miniconda3/envs/uva_1/bin/python")
    parser.add_argument("--lingbot-python", default="/local/real/jvclark/miniconda3/envs/lingbot-va/bin/python")
    parser.add_argument("--zarr-path", default="/local/real/jvclark/unified_video_action/uva/umi_data/umi_ft/WBW2-addition-wristOvershoot-harderGrasp-ft36_true2.zarr")
    parser.add_argument("--episode", type=int, default=2)
    parser.add_argument("--num-frames", type=int, default=20)
    parser.add_argument("--frame-spacing", type=int, default=12)
    parser.add_argument("--task-name", default="erase the drawing")
    parser.add_argument("--output-dir", default="/store/real/jvclark/lingbot-va/action_compare_out/step_8000")
    parser.add_argument("--uva-checkpoint", default="/store/real/jvclark/unified_video_action/checkpoints/mycheckpoints/uva_umi_multitask_video_action_vaseft_wbwnoft/checkpoints/latest.ckpt")
    parser.add_argument("--uva-device", default="cuda:0")
    parser.add_argument("--ft-normalizer-checkpoint", default=None)
    parser.add_argument("--estimate-ft", action="store_true")
    parser.add_argument("--skip-uva", action="store_true")
    parser.add_argument("--skip-lingbot", action="store_true")
    parser.add_argument("--lingbot-config-name", default="umi_wbw_train")
    parser.add_argument("--lingbot-base-root", default="/store/real/jvclark/lingbot-va/checkpoints/lingbot-va-base")
    parser.add_argument("--lingbot-checkpoint-root", default="/store/real/jvclark/lingbot-va/train_out/checkpoints/checkpoint_step_8000")
    parser.add_argument("--lingbot-device", default="cuda:0")
    parser.add_argument("--lingbot-history-stride", type=int, default=4)
    parser.add_argument("--lingbot-max-history-frames", type=int, default=16)
    parser.add_argument("--future-offsets", type=int, nargs=4, default=[3, 6, 9, 12],
                        help="Raw zarr timestep offsets to compare from the current timestep.")
    parser.add_argument("--lingbot-sample-indices", type=int, nargs=4, default=None,
                        help="Deprecated fixed LingBot chunk indices. Prefer --future-offsets.")
    parser.add_argument("--chunk-plot-frames", type=int, nargs="*", default=[0, 5, 10])
    return parser.parse_args()


def subprocess_args(args, mode):
    skip = {
        "mode",
        "uva_python",
        "lingbot_python",
        "skip_uva",
        "skip_lingbot",
    }
    argv = [str(Path(__file__).resolve()), "--mode", mode]
    pairs = vars(args)
    for key, value in pairs.items():
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
    import subprocess

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    commands = [
        [args.uva_python] + subprocess_args(args, "uva"),
        [args.lingbot_python] + subprocess_args(args, "lingbot"),
        [args.lingbot_python] + subprocess_args(args, "plot"),
    ]
    for cmd in commands:
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


def load_datapoints(args):
    from extract_episodes_with_frames import ZarrEpisodeExtractor

    extractor = ZarrEpisodeExtractor(
        zarr_path=args.zarr_path,
        n_frames_history=4,
        n_frames_output=4,
        down_sample_steps=3,
        use_relative_pose=True,
        use_ft=True,
    )

    episode_start = int(extractor.episode_starts[args.episode])
    episode_end = int(extractor.episode_ends[args.episode])
    episode_len = episode_end - episode_start
    timesteps = [i * args.frame_spacing for i in range(args.num_frames)]
    timesteps = [t for t in timesteps if t < episode_len]

    datapoints = []
    for timestep in timesteps:
        dp = extractor.load_datapoint(args.episode, timestep)
        dp["episode_idx"] = args.episode
        datapoints.append(dp)
    return extractor, timesteps, datapoints


def main():
    args = parse_args()
    if args.mode == "all":
        run_all(args)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor, timesteps, datapoints = load_datapoints(args)
    gt = np.asarray([dp["actions_relative"] for dp in datapoints])

    if args.mode == "uva":
        uva = run_uva_predictions(
            datapoints,
            checkpoint=args.uva_checkpoint,
            device=args.uva_device,
            task_name=args.task_name,
            estimate_ft=args.estimate_ft,
            ft_normalizer_checkpoint=args.ft_normalizer_checkpoint,
        )
        np.savez_compressed(
            output_dir / "uva_predictions.npz",
            timesteps=np.asarray(timesteps),
            gt_uva_pose10d=gt,
            uva_pred_pose10d=uva,
        )
        print(f"Saved UVA predictions to: {(output_dir / 'uva_predictions.npz').resolve()}")
        return

    if args.mode == "lingbot":
        lingbot, lingbot_full, lingbot_sample_ids, lingbot_pred_starts = run_lingbot_predictions(
            datapoints,
            extractor,
            args.episode,
            args,
        )
        np.savez_compressed(
            output_dir / "lingbot_predictions.npz",
            timesteps=np.asarray(timesteps),
            gt_uva_pose10d=gt,
            lingbot_pred_pose10d=lingbot,
            lingbot_full_pose10d=lingbot_full,
            lingbot_full_16_pose10d=lingbot_full,
            lingbot_sample_indices=lingbot_sample_ids,
            lingbot_pred_start_timesteps=lingbot_pred_starts,
            future_offsets=np.asarray(args.future_offsets),
        )
        print(f"Saved LingBot predictions to: {(output_dir / 'lingbot_predictions.npz').resolve()}")
        return

    uva_path = output_dir / "uva_predictions.npz"
    lingbot_path = output_dir / "lingbot_predictions.npz"
    if not uva_path.exists() or not lingbot_path.exists():
        raise FileNotFoundError(
            f"Plot mode needs {uva_path} and {lingbot_path}. Run --mode all, or run --mode uva and --mode lingbot first."
        )
    uva_npz = np.load(uva_path)
    lingbot_npz = np.load(lingbot_path)
    uva = uva_npz["uva_pred_pose10d"]
    lingbot = lingbot_npz["lingbot_pred_pose10d"]
    lingbot_full = lingbot_npz["lingbot_full_pose10d"] if "lingbot_full_pose10d" in lingbot_npz.files else lingbot_npz["lingbot_full_16_pose10d"]
    lingbot_sample_ids = (
        lingbot_npz["lingbot_sample_indices"]
        if "lingbot_sample_indices" in lingbot_npz.files
        else np.tile(np.arange(4), (len(timesteps), 1))
    )
    future_offsets = (
        lingbot_npz["future_offsets"]
        if "future_offsets" in lingbot_npz.files
        else np.asarray(args.future_offsets)
    )
    lingbot_pred_starts = (
        lingbot_npz["lingbot_pred_start_timesteps"]
        if "lingbot_pred_start_timesteps" in lingbot_npz.files
        else np.full(len(timesteps), -1)
    )

    np.savez_compressed(
        output_dir / "predictions.npz",
        timesteps=np.asarray(timesteps),
        gt_uva_pose10d=gt,
        uva_pred_pose10d=uva,
        lingbot_pred_pose10d=lingbot,
        lingbot_full_pose10d=lingbot_full,
        lingbot_full_16_pose10d=lingbot_full,
        lingbot_sample_indices=lingbot_sample_ids,
        lingbot_pred_start_timesteps=lingbot_pred_starts,
        future_offsets=future_offsets,
    )

    plot_episode_timeline(
        np.asarray(timesteps),
        gt,
        uva,
        lingbot,
        output_dir / "episode_first_action_timeline.png",
    )

    for frame_idx in args.chunk_plot_frames:
        if frame_idx < 0 or frame_idx >= len(datapoints):
            continue
        plot_chunk_panel(
            datapoints[frame_idx],
            gt[frame_idx],
            uva[frame_idx],
            lingbot[frame_idx],
            lingbot_full[frame_idx],
            output_dir / f"chunk_frame_{frame_idx:03d}_t{timesteps[frame_idx]:04d}.png",
            lingbot_sample_ids=lingbot_sample_ids[frame_idx],
            future_offsets=future_offsets,
        )

    print(f"Saved comparison outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
