#!/usr/bin/env python3
"""Convert zarr frames directly into official LeRobot v3 on-disk dataset layout.

Unlike reorganize_zarr_for_lerobot_v3.py which requires pre-transcoded videos,
this script reads camera frame arrays (camera0_rgb, etc.) directly from the zarr
and encodes them to video via ffmpeg.

Output layout (LeRobot v3):
- meta/info.json
- meta/tasks.jsonl
- meta/episodes/chunk-000/file-000.parquet
- data/chunk-XXX/file-YYY.parquet
- videos/{video_key}/chunk-XXX/file-YYY.mp4

Feature naming convention:
- {side}_{cam_name}_pose_in_main_xyz_wxyz: (P,)                  Non-main camera pose in main frame
- observation.state.{side}_main_camera_trajectory_xyz_wxyz: (P,)  Main camera world-frame pose
- observation.state.{side}_eef_state_gripper_width_m: (E,)        Gripper width in meters
- observation.image.{side}_{image_name}: video                    Camera video

Constant calibration (stored once in info.json, not per-frame):
- {side}_eef_pose_in_main_xyz_wxyz: (P,)  EEF (TCP) pose in main camera frame

Where P=7 (xyz + quaternion wxyz), E=eef_state_dim (typically 1 for gripper).

Usage:
    python scripts/zarr_to_lerobot_v3.py --config example_demo_session.json
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import json
import pathlib
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import glob as glob_mod

import click
import numpy as np
import pandas as pd
import zarr
from scipy.spatial.transform import Rotation

# Keys written into info.json camera / gripper metadata.
# Keep in sync across all convert-to-lerobot scripts.
CAMERA_META_KEYS = ("model", "raw_res", "is_fisheye")
GRIPPER_META_KEYS = (
    "gripper_model",
    "gripper_width_m_max",
    "gripper_width_m_min",
    "gripper_units",
    "gripper_native_min",
    "gripper_native_max",
)


class OnlineStats:
    """Streaming mean/std/min/max via Chan's parallel update.

    O(dim) memory per key, independent of the total frame count. ``update``
    accepts a ``(N, dim)`` (or 1-D, promoted to ``(N, 1)``) batch; ``finalize``
    returns a dict with lerobot's stats schema.
    Duplicated across convert-to-lerobot scripts — keep in sync.
    """

    __slots__ = ("n", "mean", "M2", "min", "max")

    def __init__(self) -> None:
        self.n = 0
        self.mean = None
        self.M2 = None
        self.min = None
        self.max = None

    def update(self, batch: np.ndarray) -> None:
        flat = batch.reshape(batch.shape[0], -1) if batch.ndim > 1 else batch[:, None]
        n_b = flat.shape[0]
        if n_b == 0:
            return
        m_b = flat.mean(axis=0)
        M2_b = ((flat - m_b) ** 2).sum(axis=0)
        mn_b = flat.min(axis=0)
        mx_b = flat.max(axis=0)
        if self.n == 0:
            self.mean, self.M2 = m_b, M2_b
            self.min, self.max = mn_b, mx_b
        else:
            delta = m_b - self.mean
            N = self.n + n_b
            self.mean = self.mean + delta * (n_b / N)
            self.M2 = self.M2 + M2_b + (delta ** 2) * (self.n * n_b / N)
            self.min = np.minimum(self.min, mn_b)
            self.max = np.maximum(self.max, mx_b)
        self.n += n_b

    def finalize(self) -> Dict[str, list]:
        std = np.sqrt(self.M2 / self.n) if self.n > 0 else np.zeros_like(self.mean)
        return {
            "mean": self.mean.tolist(),
            "std": std.tolist(),
            "min": self.min.tolist(),
            "max": self.max.tolist(),
        }


def _convert_gripper_units(arm: Dict, state_arr: np.ndarray) -> np.ndarray:
    """Linearly remap raw eef state values to gripper_width_m range.

    If ``gripper_units`` is set (and not ``"m"``), the raw values are assumed
    to live in ``[gripper_native_min, gripper_native_max]`` (per-key) and are
    linearly mapped onto ``[gripper_width_m_min, gripper_width_m_max]``,
    clamping to the native range first. Otherwise the input is returned
    unchanged.
    """
    units = arm.get("gripper_units")
    if not units or units == "m":
        return state_arr
    nmin = np.asarray(arm["gripper_native_min"], dtype=np.float32)
    nmax = np.asarray(arm["gripper_native_max"], dtype=np.float32)
    gmin = np.asarray(arm["gripper_width_m_min"], dtype=np.float32)
    gmax = np.asarray(arm["gripper_width_m_max"], dtype=np.float32)
    clamped = np.clip(state_arr, nmin, nmax)
    return (((clamped - nmin) / (nmax - nmin)) * (gmax - gmin) + gmin).astype(np.float32)

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs

register_codecs()

CAMERA_RGB_RE = re.compile(r"^camera\d+_rgb$")
POSE_DIM = 7  # xyz + quaternion wxyz


def load_config_file(config_file: str) -> dict:
    config_path = config_file
    if not os.path.isabs(config_path):
        config_path = os.path.join(ROOT_DIR, "config", config_file)
    with open(config_path, "r") as f:
        return json.load(f)


def resolve_path(path_str: str) -> pathlib.Path:
    path = pathlib.Path(path_str).expanduser()
    if not path.is_absolute():
        path = pathlib.Path(os.path.abspath(os.path.join(ROOT_DIR, path_str)))
    return path


def open_rb(path: pathlib.Path, mode: str):
    if path.suffix == ".zip":
        store = zarr.ZipStore(str(path), mode=mode)
        root = zarr.group(store=store)
        return store, ReplayBuffer.create_from_group(root)
    root = zarr.open(str(path), mode=mode)
    return None, ReplayBuffer.create_from_group(root)


def _episode_slices(episode_ends: np.ndarray) -> List[slice]:
    starts = np.concatenate([[0], episode_ends[:-1]])
    return [slice(int(s), int(e)) for s, e in zip(starts, episode_ends)]


# ---------------------------------------------------------------------------
# Video encoding from zarr frames
# ---------------------------------------------------------------------------

def _encode_frames_to_video(
    frames: np.ndarray,
    output_path: pathlib.Path,
    fps: float,
) -> None:
    """Encode (N, H, W, 3) uint8 numpy frames to an mp4 video via ffmpeg pipe.

    Uses GPU (av1_nvenc) and raises if encoding fails.
    Writes raw frame bytes via stdin pipe for maximum throughput.
    """
    n, h, w, c = frames.shape
    assert c == 3, f"Expected 3-channel RGB frames, got {c}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_bytes = frames.tobytes()

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "av1_nvenc",
        "-cq", "30", "-b:v", "0", "-preset", "p5",
        "-pix_fmt", "yuv420p",
        "-an",
        str(output_path),
    ]
    result = subprocess.run(cmd, input=raw_bytes, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"av1_nvenc encoding failed for {output_path}: {result.stderr.decode()}"
        )


# ---------------------------------------------------------------------------
# Pose helpers
# ---------------------------------------------------------------------------

def _rotvec_to_quat_wxyz(rotvec: np.ndarray) -> np.ndarray:
    """Convert rotation vector(s) to quaternion [w, x, y, z] format."""
    q_xyzw = Rotation.from_rotvec(rotvec).as_quat()
    if q_xyzw.ndim == 1:
        return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)
    return np.column_stack([q_xyzw[:, 3], q_xyzw[:, 0], q_xyzw[:, 1], q_xyzw[:, 2]]).astype(np.float32)


def _T_to_pose7(T: np.ndarray) -> np.ndarray:
    """Convert (4,4) or (N,4,4) transform(s) to [x, y, z, qw, qx, qy, qz]."""
    if T.ndim == 2:
        pos = T[:3, 3]
        quat = _rotvec_to_quat_wxyz(Rotation.from_matrix(T[:3, :3]).as_rotvec())
        return np.concatenate([pos, quat]).astype(np.float32)
    pos = T[:, :3, 3]
    rotvecs = Rotation.from_matrix(T[:, :3, :3]).as_rotvec()
    quats = _rotvec_to_quat_wxyz(rotvecs)
    return np.concatenate([pos, quats], axis=1).astype(np.float32)


def _build_T_cam_in_tcp(translation: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """Build 4x4 T_cam_to_tcp from camera_in_tcp translation and quaternion wxyz."""
    T = np.eye(4, dtype=np.float32)
    q_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
    T[:3, :3] = Rotation.from_quat(q_xyzw).as_matrix().astype(np.float32)
    T[:3, 3] = translation
    return T


def _compute_T_world(pos: np.ndarray, rot_aa: np.ndarray) -> np.ndarray:
    """Compute (N, 4, 4) world-frame transforms from position + axis-angle."""
    n = len(pos)
    mats = Rotation.from_rotvec(rot_aa).as_matrix().astype(np.float32)
    T = np.zeros((n, 4, 4), dtype=np.float32)
    T[:, 3, 3] = 1.0
    T[:, :3, :3] = mats
    T[:, :3, 3] = pos
    return T


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_arm_config(entries: list) -> List[Dict]:
    """Load arm configuration.

    Expected format per entry::

        {
            "side": "right",
            "robot_name": "robot0",
            "eef_state_keys": ["robot0_gripper_width"],
            "cameras": [
                {
                    "role": "main",
                    "image_name": "main_camera_rgb",
                    "camera_index": 0,
                    "camera_in_tcp": {
                        "translation": [x, y, z],
                        "rotation_quat_wxyz": [qw, qx, qy, qz]
                    }
                }
            ]
        }
    """
    arms = []
    for entry in entries:
        side = entry["side"]
        robot_name = entry["robot_name"]
        eef_state_keys = entry.get("eef_state_keys", [f"{robot_name}_gripper_width"])
        cameras = []
        for cam in entry["cameras"]:
            camera_in_tcp = cam.get("camera_in_tcp", {})
            translation = np.asarray(camera_in_tcp.get("translation", [0, 0, 0]), dtype=np.float32)
            quat_wxyz = np.asarray(camera_in_tcp.get("rotation_quat_wxyz", [1, 0, 0, 0]), dtype=np.float32)
            T_cam_to_tcp = _build_T_cam_in_tcp(translation, quat_wxyz)
            T_tcp_to_cam = np.linalg.inv(T_cam_to_tcp).astype(np.float32)
            image_name = cam.get("image_name", f"{cam['role']}_camera_rgb")
            cam_entry = {
                "role": cam["role"],
                "image_name": image_name,
                "camera_index": cam["camera_index"],
                "T_cam_to_tcp": T_cam_to_tcp,
                "T_tcp_to_cam": T_tcp_to_cam,
            }
            for meta_key in CAMERA_META_KEYS:
                if meta_key in cam:
                    cam_entry[meta_key] = cam[meta_key]
            cameras.append(cam_entry)
        arm_entry = {
            "side": side,
            "robot_name": robot_name,
            "eef_pos_key": f"{robot_name}_eef_pos",
            "eef_rot_key": f"{robot_name}_eef_rot_axis_angle",
            "eef_state_keys": eef_state_keys,
            "cameras": cameras,
        }
        for gk in GRIPPER_META_KEYS:
            if gk in entry:
                arm_entry[gk] = entry[gk]
        arms.append(arm_entry)
    return arms


# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------

def _build_features(
    arms: List[Dict],
    video_shapes: Dict[int, tuple],
) -> Dict[str, dict]:
    """Build the LeRobot v3 features dict from arm config and probed video shapes."""
    features = {
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }

    for arm in arms:
        side = arm["side"]

        # Main camera world-frame trajectory
        traj_key = f"observation.state.{side}_main_camera_trajectory_xyz_wxyz"
        features[traj_key] = {
            "dtype": "float32", "shape": [POSE_DIM],
            "names": [f"{traj_key}_{i}" for i in range(POSE_DIM)],
        }

        # EEF state (e.g. gripper width)
        E = len(arm["eef_state_keys"])
        state_key = f"observation.state.{side}_gripper_width_m"
        features[state_key] = {
            "dtype": "float32", "shape": [E],
            "names": [f"{state_key}_{i}" for i in range(E)],
        }

        for cam in arm["cameras"]:
            # Non-main camera pose relative to main camera
            if cam["role"] != "main":
                pose_name = cam["image_name"].removesuffix("_rgb")
                pose_key = f"{side}_{pose_name}_pose_in_main_xyz_wxyz"
                features[pose_key] = {
                    "dtype": "float32", "shape": [POSE_DIM],
                    "names": [f"{pose_key}_{i}" for i in range(POSE_DIM)],
                }

            # Video feature — shape from zarr frame array
            vid_key = f"observation.image.{side}_{cam['image_name']}"
            cam_idx = cam["camera_index"]
            if cam_idx in video_shapes:
                h, w, c = video_shapes[cam_idx]
                features[vid_key] = {
                    "dtype": "video", "shape": [h, w, c],
                    "names": ["height", "width", "channels"],
                }

    return features


# ---------------------------------------------------------------------------
# Shard merging (same as reorganize_zarr_for_lerobot_v3.py)
# ---------------------------------------------------------------------------

def _group_by_size(episode_indices, size_fn, target_bytes):
    """Group consecutive episodes into batches whose cumulative size <= target_bytes."""
    batches = []
    current = []
    current_size = 0
    for ep_idx in episode_indices:
        ep_size = size_fn(ep_idx)
        if current and current_size + ep_size > target_bytes:
            batches.append(current)
            current = [ep_idx]
            current_size = ep_size
        else:
            current.append(ep_idx)
            current_size += ep_size
    if current:
        batches.append(current)
    return batches


def _merge_episode_shards(dataset_root: str,
                          data_files_size_in_mb: float = 100.0,
                          video_files_size_in_mb: float = 200.0):
    """Merge per-episode shards into multi-episode files."""
    import os as _os

    dataset_root = _os.path.abspath(dataset_root)
    with open(_os.path.join(dataset_root, "meta", "info.json")) as f:
        info = json.load(f)

    fps = float(info["fps"])
    video_template = info["video_path"]
    data_template = info["data_path"]
    features = info.get("features", {})
    video_keys = sorted(k for k, v in features.items() if v.get("dtype") == "video")

    ep_meta_files = sorted(glob_mod.glob(_os.path.join(dataset_root, "meta", "episodes", "chunk-*", "file-*.parquet")))
    if not ep_meta_files:
        return

    ep_rows = []
    for fpath in ep_meta_files:
        df = pd.read_parquet(fpath)
        for _, row in df.iterrows():
            ep_rows.append(dict(row))
    ep_rows.sort(key=lambda r: int(r["episode_index"]))
    ep_row_map = {int(r["episode_index"]): r for r in ep_rows}
    episode_indices = [int(r["episode_index"]) for r in ep_rows]
    ep_length_map = {int(r["episode_index"]): int(r["length"]) for r in ep_rows}

    ep_video_paths = {}
    ep_data_paths = {}
    ep_data_sizes = {}
    ep_video_sizes = {}

    for row in ep_rows:
        ep_idx = int(row["episode_index"])
        ep_video_paths[ep_idx] = {}
        max_vid_size = 0
        for vk in video_keys:
            vid_path = _os.path.join(dataset_root, video_template.format(
                video_key=vk,
                chunk_index=int(row[f"videos/{vk}/chunk_index"]),
                file_index=int(row[f"videos/{vk}/file_index"]),
            ))
            ep_video_paths[ep_idx][vk] = vid_path
            sz = _os.path.getsize(vid_path) if _os.path.isfile(vid_path) else 0
            max_vid_size = max(max_vid_size, sz)
        ep_video_sizes[ep_idx] = max_vid_size

        data_path = _os.path.join(dataset_root, data_template.format(
            chunk_index=int(row["data/chunk_index"]),
            file_index=int(row["data/file_index"]),
        ))
        ep_data_paths[ep_idx] = data_path
        ep_data_sizes[ep_idx] = _os.path.getsize(data_path) if _os.path.isfile(data_path) else 0

    data_batches = _group_by_size(episode_indices, lambda e: ep_data_sizes[e],
                                  int(data_files_size_in_mb * 1024 * 1024))
    video_batches = _group_by_size(episode_indices, lambda e: ep_video_sizes[e],
                                   int(video_files_size_in_mb * 1024 * 1024))

    print(f"  merge: {len(episode_indices)} episodes -> "
          f"{len(data_batches)} data shards, {len(video_batches)} video shards")

    for batch_idx, batch in enumerate(data_batches):
        chunk_idx = batch_idx // 1000
        file_idx = batch_idx % 1000
        for ep_idx in batch:
            row = ep_row_map[ep_idx]
            row["data/chunk_index"] = chunk_idx
            row["data/file_index"] = file_idx
            row["meta/episodes/chunk_index"] = chunk_idx
            row["meta/episodes/file_index"] = file_idx

    for batch_idx, batch in enumerate(video_batches):
        chunk_idx = batch_idx // 1000
        file_idx = batch_idx % 1000
        running_time = 0.0
        for ep_idx in batch:
            row = ep_row_map[ep_idx]
            ep_duration = float(ep_length_map[ep_idx]) / fps
            for vk in video_keys:
                row[f"videos/{vk}/chunk_index"] = chunk_idx
                row[f"videos/{vk}/file_index"] = file_idx
                row[f"videos/{vk}/from_timestamp"] = running_time
                row[f"videos/{vk}/to_timestamp"] = running_time + ep_duration
            running_time += ep_duration

    # -- Merge videos --
    old_video_files = set()
    for ep_idx in episode_indices:
        for vk in video_keys:
            old_video_files.add(_os.path.abspath(ep_video_paths[ep_idx][vk]))

    for batch_idx, batch in enumerate(video_batches):
        new_chunk = batch_idx // 1000
        new_file = batch_idx % 1000
        for vk in video_keys:
            merged_dir = _os.path.join(dataset_root, "videos", vk, f"chunk-{new_chunk:03d}")
            _os.makedirs(merged_dir, exist_ok=True)
            merged_path = _os.path.join(merged_dir, f"file-{new_file:03d}.mp4")

            if len(batch) == 1:
                src = ep_video_paths[batch[0]][vk]
                if _os.path.abspath(src) != _os.path.abspath(merged_path):
                    _os.rename(src, merged_path)
                old_video_files.discard(_os.path.abspath(src))
                old_video_files.discard(_os.path.abspath(merged_path))
                continue

            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
                for ep_idx in batch:
                    tmp.write(f"file '{ep_video_paths[ep_idx][vk]}'\n")
                concat_list = tmp.name

            tmp_merged = merged_path + ".tmp.mp4"
            try:
                cmd = [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "concat", "-safe", "0",
                    "-i", concat_list, "-c", "copy", tmp_merged,
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"ffmpeg concat failed for {vk} batch {batch_idx}: {result.stderr}")
                _os.rename(tmp_merged, merged_path)
            finally:
                if _os.path.isfile(concat_list):
                    _os.unlink(concat_list)
                if _os.path.isfile(tmp_merged):
                    _os.unlink(tmp_merged)

            old_video_files.discard(_os.path.abspath(merged_path))

    for f in old_video_files:
        if _os.path.isfile(f):
            _os.unlink(f)

    # -- Merge data parquets --
    old_data_files = set(_os.path.abspath(ep_data_paths[e]) for e in episode_indices)
    for batch_idx, batch in enumerate(data_batches):
        new_chunk = batch_idx // 1000
        new_file = batch_idx % 1000
        merged_data_path = _os.path.join(
            dataset_root, "data", f"chunk-{new_chunk:03d}", f"file-{new_file:03d}.parquet",
        )
        _os.makedirs(_os.path.dirname(merged_data_path), exist_ok=True)
        dfs = [pd.read_parquet(ep_data_paths[ep_idx]) for ep_idx in batch]
        pd.concat(dfs, ignore_index=True).to_parquet(merged_data_path)
        old_data_files.discard(_os.path.abspath(merged_data_path))

    for f in old_data_files:
        if _os.path.isfile(f):
            _os.unlink(f)

    # -- Rewrite episode metadata (follows data chunking) --
    for fpath in ep_meta_files:
        if _os.path.isfile(fpath):
            _os.unlink(fpath)

    for batch_idx, batch in enumerate(data_batches):
        new_chunk = batch_idx // 1000
        new_file = batch_idx % 1000
        batch_rows = [ep_row_map[ep_idx] for ep_idx in batch]
        ep_meta_path = _os.path.join(
            dataset_root, "meta", "episodes",
            f"chunk-{new_chunk:03d}", f"file-{new_file:03d}.parquet",
        )
        _os.makedirs(_os.path.dirname(ep_meta_path), exist_ok=True)
        pd.DataFrame(batch_rows).to_parquet(ep_meta_path)

    # Clean up empty directories
    for subdir in ["videos", "data", _os.path.join("meta", "episodes")]:
        base = _os.path.join(dataset_root, subdir)
        if subdir == "videos":
            for vk in video_keys:
                vk_dir = _os.path.join(base, vk)
                for d in sorted(glob_mod.glob(_os.path.join(vk_dir, "chunk-*"))):
                    if _os.path.isdir(d) and not _os.listdir(d):
                        _os.rmdir(d)
        else:
            for d in sorted(glob_mod.glob(_os.path.join(base, "chunk-*"))):
                if _os.path.isdir(d) and not _os.listdir(d):
                    _os.rmdir(d)

    print(f"  merge: done")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@click.command()
@click.option("--config", "config_file", required=True, type=str,
              help="JSON config file (absolute or relative to config/)")
@click.option("--num-workers", "num_workers", default=None, type=int,
              help="Max parallel av1_nvenc workers (default: 2)")
def main(config_file: str, num_workers: Optional[int]):
    loaded_config = load_config_file(config_file)
    dpc = loaded_config.get("data_process_config", {})
    ds = loaded_config.get("device_settings", {})

    input_path = dpc.get("input_zarr")
    output_dir = dpc.get("lerobot_output_dir")
    repo_id = dpc.get("lerobot_repo_id", "local/umi_dataset")
    overwrite = bool(dpc.get("lerobot_overwrite", False))
    tasks = dpc.get("lerobot_tasks", ["default"])
    data_files_size_in_mb = float(dpc.get("data_files_size_in_mb", 100))
    video_files_size_in_mb = float(dpc.get("video_files_size_in_mb", 200))
    robot_type = ds.get("robot_type")
    fps = float(dpc["fps"])
    arm_config_entries = dpc.get("arm_config", [])

    if input_path is None:
        raise click.ClickException("config.data_process_config.input_zarr is required")
    if output_dir is None:
        raise click.ClickException("config.data_process_config.lerobot_output_dir is required")
    if not arm_config_entries:
        raise click.ClickException("config.data_process_config.arm_config is required")

    # Support both single string and list of strings for multi-zarr configs
    input_paths = input_path if isinstance(input_path, list) else [input_path]

    out_root = resolve_path(output_dir)
    arms = _load_arm_config(arm_config_entries)

    if out_root.exists() and any(out_root.iterdir()):
        if not overwrite:
            raise click.ClickException(
                f"Output dir not empty: {out_root}. Set lerobot_overwrite to true in config."
            )
        shutil.rmtree(out_root)

    (out_root / "meta" / "episodes").mkdir(parents=True, exist_ok=True)
    (out_root / "data").mkdir(parents=True, exist_ok=True)
    (out_root / "videos").mkdir(parents=True, exist_ok=True)

    # Build camera_index -> video key mapping.
    cam_idx_to_vid_key: Dict[int, str] = {}
    for arm in arms:
        for cam in arm["cameras"]:
            cam_idx_to_vid_key[cam["camera_index"]] = (
                f"observation.image.{arm['side']}_{cam['image_name']}"
            )

    global_index = 0
    ep_idx = 0
    total_episodes = 0
    features = None
    task_to_index: Dict[str, int] = {}
    # Streaming per-key stats. Only data keys go in here — index columns
    # (timestamp/frame_index/...) are never added, mirroring the genrobot /
    # fast_umi conversion scripts.
    stats_buffers: Dict[str, OnlineStats] = {}

    # av1_nvenc is NVENC-session-bound on consumer GeForce.
    max_workers = num_workers if num_workers is not None else 2
    print(f"Using {max_workers} parallel video encoding workers")

    for zarr_idx, ip in enumerate(input_paths):
        src_path = resolve_path(ip)
        if len(input_paths) > 1:
            print(f"\n{'='*60}")
            print(f"Processing zarr {zarr_idx+1}/{len(input_paths)}: {src_path}")
            print(f"{'='*60}")

        src_store, src_rb = open_rb(src_path, mode="r")
        try:
            episode_ends = src_rb.episode_ends[:]
            ep_slices = _episode_slices(episode_ends)
            n_episodes = len(ep_slices)

            if n_episodes == 0:
                print(f"Warning: zarr {src_path} has zero episodes, skipping.")
                continue

            # Discover camera RGB arrays in the zarr.
            cam_indices = []
            for key in src_rb.data.keys():
                if CAMERA_RGB_RE.match(key):
                    idx = int(key[len("camera"):-len("_rgb")])
                    cam_indices.append(idx)
            cam_indices = sorted(cam_indices)
            if not cam_indices:
                raise click.ClickException("No camera*_rgb arrays found in zarr data")

            # Determine video shapes from zarr frame arrays.
            video_shapes: Dict[int, tuple] = {}
            for cam_idx in cam_indices:
                arr = src_rb.data[f"camera{cam_idx}_rgb"]
                # arr shape is (T, H, W, C)
                _, h, w, c = arr.shape
                video_shapes[cam_idx] = (h, w, c)
                print(f"Camera {cam_idx} frame shape from zarr: ({h}, {w}, {c})")

            # Set raw_res on each camera entry from the zarr frame resolution.
            for arm in arms:
                for cam in arm["cameras"]:
                    ci = cam["camera_index"]
                    if ci in video_shapes:
                        cam["raw_res"] = video_shapes[ci][:2]

            if features is None:
                features = _build_features(arms, video_shapes)

            # Process episodes with parallel video encoding.
            # Strategy: read zarr frames (I/O bound) and submit encode jobs to thread pool.
            # Parquet writes happen in the main thread (cheap), video encodes run in parallel.
            encode_futures = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for src_ep_idx, sl in enumerate(ep_slices):
                    ep_len = int(sl.stop - sl.start)
                    chunk_idx = ep_idx // 1000
                    file_idx = ep_idx % 1000

                    print(f"  Episode {ep_idx} (source {src_ep_idx}): {ep_len} frames")

                    data_abs = out_root / "data" / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.parquet"
                    data_abs.parent.mkdir(parents=True, exist_ok=True)

                    frame_index = np.arange(ep_len, dtype=np.int64)
                    timestamp = frame_index.astype(np.float32) / float(fps)
                    index = np.arange(global_index, global_index + ep_len, dtype=np.int64)

                    # Map each task name to a unique index.
                    task_name = tasks[0] if tasks else "default"
                    if task_name not in task_to_index:
                        task_to_index[task_name] = len(task_to_index)

                    data_dict = {
                        "timestamp": timestamp,
                        "frame_index": frame_index,
                        "episode_index": np.full(ep_len, ep_idx, dtype=np.int64),
                        "index": index,
                        "task_index": np.full(ep_len, task_to_index[task_name], dtype=np.int64),
                    }

                    # --- Per-arm computed features ---
                    for arm in arms:
                        side = arm["side"]
                        eef_pos = np.asarray(src_rb.data[arm["eef_pos_key"]][sl], dtype=np.float32)
                        eef_rot_aa = np.asarray(src_rb.data[arm["eef_rot_key"]][sl], dtype=np.float32)

                        # World-frame EEF transform: (N, 4, 4)
                        T_world_eef = _compute_T_world(eef_pos, eef_rot_aa)

                        # Compute world-frame transform for each camera.
                        T_world_main = None
                        cam_world_Ts: Dict[str, np.ndarray] = {}
                        for cam in arm["cameras"]:
                            T_world_cam = T_world_eef @ cam["T_cam_to_tcp"]
                            cam_world_Ts[cam["role"]] = T_world_cam
                            if cam["role"] == "main":
                                T_world_main = T_world_cam

                        if T_world_main is None:
                            raise click.ClickException(f"No camera with role='main' for side '{side}'")

                        T_main_inv = np.linalg.inv(T_world_main)

                        # Main camera world-frame trajectory
                        traj_key = f"observation.state.{side}_main_camera_trajectory_xyz_wxyz"
                        traj_arr = _T_to_pose7(T_world_main).astype(np.float32)
                        data_dict[traj_key] = [x.tolist() for x in traj_arr]
                        stats_buffers.setdefault(traj_key, OnlineStats()).update(traj_arr)

                        # Non-main camera poses relative to main camera
                        for cam in arm["cameras"]:
                            if cam["role"] == "main":
                                continue
                            T_main_cam = T_main_inv @ cam_world_Ts[cam["role"]]
                            pose_name = cam["image_name"].removesuffix("_rgb")
                            pose_key = f"{side}_{pose_name}_pose_in_main_xyz_wxyz"
                            pose_arr = _T_to_pose7(T_main_cam).astype(np.float32)
                            data_dict[pose_key] = [x.tolist() for x in pose_arr]
                            stats_buffers.setdefault(pose_key, OnlineStats()).update(pose_arr)

                        # EEF state (e.g. gripper width)
                        state_key = f"observation.state.{side}_gripper_width_m"
                        state_arrays = []
                        for sk in arm["eef_state_keys"]:
                            arr = np.asarray(src_rb.data[sk][sl], dtype=np.float32)
                            if arr.ndim == 1:
                                arr = arr[:, None]
                            state_arrays.append(arr)
                        state_arr = np.concatenate(state_arrays, axis=1)
                        state_arr = _convert_gripper_units(arm, state_arr)
                        if state_arr.shape[1] == 1:
                            data_dict[state_key] = state_arr[:, 0]
                        else:
                            data_dict[state_key] = [x.tolist() for x in state_arr]
                        stats_buffers.setdefault(state_key, OnlineStats()).update(state_arr)

                    pd.DataFrame(data_dict).to_parquet(data_abs, engine="pyarrow")

                    # --- Read zarr frames and submit video encodes in parallel ---
                    ep_row = {
                        "episode_index": ep_idx,
                        "tasks": json.dumps(tasks),
                        "length": ep_len,
                        "dataset_from_index": int(global_index),
                        "dataset_to_index": int(global_index + ep_len),
                        "data/chunk_index": int(chunk_idx),
                        "data/file_index": int(file_idx),
                        "meta/episodes/chunk_index": int(chunk_idx),
                        "meta/episodes/file_index": int(file_idx),
                    }

                    for cam_idx in cam_indices:
                        vid_key = cam_idx_to_vid_key.get(cam_idx)
                        if vid_key is None:
                            continue

                        # Read frames from zarr (I/O bound — do in main thread before submit)
                        frames = np.asarray(src_rb.data[f"camera{cam_idx}_rgb"][sl], dtype=np.uint8)

                        vid_rel = (
                            pathlib.Path("videos") / vid_key
                            / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
                        )
                        vid_abs = out_root / vid_rel

                        # Submit encoding to thread pool (CPU/GPU bound)
                        fut = executor.submit(
                            _encode_frames_to_video, frames, vid_abs, fps,
                        )
                        encode_futures.append((fut, ep_idx, vid_key, str(vid_abs)))

                        ep_row[f"videos/{vid_key}/chunk_index"] = int(chunk_idx)
                        ep_row[f"videos/{vid_key}/file_index"] = int(file_idx)
                        ep_row[f"videos/{vid_key}/from_timestamp"] = 0.0
                        ep_row[f"videos/{vid_key}/to_timestamp"] = float(ep_len) / float(fps)

                    episodes_path = (
                        out_root / "meta" / "episodes"
                        / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.parquet"
                    )
                    episodes_path.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame([ep_row]).to_parquet(episodes_path, engine="pyarrow")
                    global_index += ep_len
                    ep_idx += 1

                # Wait for all video encodes to finish and propagate errors.
                for fut, eidx, vk, vpath in encode_futures:
                    fut.result()  # raises if encoding failed
                print(f"All video encodes completed for {src_path}")

            total_episodes += n_episodes
            print(f"Processed {n_episodes} episodes from {src_path}")
        finally:
            if src_store is not None:
                src_store.close()

    n_episodes = total_episodes

    if n_episodes == 0:
        raise click.ClickException("No episodes found across all input zarrs.")

    # --- Task table ---
    with open(out_root / "meta" / "tasks.jsonl", "w") as _tf:
        for t, ti in sorted(task_to_index.items(), key=lambda x: x[1]):
            row = {"task_index": ti, "task": t}
            _tf.write(json.dumps(row) + "\n")

    # --- Constant calibration poses ---
    calibration = {}
    for arm in arms:
        side = arm["side"]
        for cam in arm["cameras"]:
            if cam["role"] == "main":
                eef_key = f"{side}_eef_pose_in_main_xyz_wxyz"
                calibration[eef_key] = _T_to_pose7(cam["T_tcp_to_cam"]).tolist()
                break

    # --- Camera metadata from config ---
    def _build_video_pipeline(cam: dict) -> str:
        """Build ffmpeg pipeline string matching the actual pipe-based encode."""
        rr = cam.get("raw_res")
        size_arg = f"{rr[1]}x{rr[0]}" if rr else "{width}x{height}"
        return (
            f"ffmpeg -f rawvideo -pix_fmt rgb24 -s {size_arg} -r {fps} -i pipe:0"
            " -c:v av1_nvenc -cq 30 -b:v 0 -preset p5 -pix_fmt yuv420p -an {output}"
        )

    camera_meta = {}
    for arm in arms:
        side = arm["side"]
        seen_prefixes = set()
        for cam in arm["cameras"]:
            if cam["role"] == "stereo":
                meta_prefix = f"{side}_stereo_camera"
            else:
                meta_prefix = f"{side}_{cam['image_name'].removesuffix('_rgb')}"

            if meta_prefix in seen_prefixes:
                continue
            seen_prefixes.add(meta_prefix)

            for meta_key in CAMERA_META_KEYS:
                if meta_key in cam:
                    camera_meta[f"{meta_prefix}_{meta_key}"] = cam[meta_key]

            camera_meta[f"{meta_prefix}_video_pipeline"] = _build_video_pipeline(cam)

    # --- Gripper metadata from config ---
    gripper_meta = {}
    for arm in arms:
        side = arm["side"]
        for meta_key in GRIPPER_META_KEYS:
            if meta_key in arm:
                gripper_meta[f"{side}_{meta_key}"] = arm[meta_key]

    # --- info.json ---
    info = {
        "codebase_version": "v3.0",
        "robot_type": robot_type,
        "robot_setup_type": ds.get("robot_setup_type", "single_arm"),
        "include_head_camera": ds.get("include_head_camera", False),
        "include_body_camera": ds.get("include_body_camera", False),
        "third_camera_num": ds.get("third_camera_num", 0),
        "has_mirrors": ds.get("has_mirrors", False),
        "total_episodes": int(n_episodes),
        "total_frames": int(global_index),
        "total_tasks": len(tasks),
        "data_files_size_in_mb": data_files_size_in_mb,
        "video_files_size_in_mb": video_files_size_in_mb,
        "fps": float(fps),
        "splits": {"train": f"0:{n_episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
        **calibration,
        **camera_meta,
        **gripper_meta,
        "repo_id": repo_id,
    }
    (out_root / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    # --- stats.json ---
    # Stats were streamed over lowdim arrays during episode processing, so
    # index columns (timestamp/frame_index/...) are naturally excluded.
    stats = {key: stats_buffers[key].finalize() for key in sorted(stats_buffers)}
    (out_root / "meta" / "stats.json").write_text(json.dumps(stats, indent=2))

    # --- Merge per-episode shards ---
    _merge_episode_shards(str(out_root), data_files_size_in_mb=data_files_size_in_mb,
                          video_files_size_in_mb=video_files_size_in_mb)

    print(f"Converted {n_episodes} episodes to LeRobot v3 format at: {out_root}")


if __name__ == "__main__":
    main()
