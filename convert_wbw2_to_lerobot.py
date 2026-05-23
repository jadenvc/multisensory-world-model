#!/usr/bin/env python3
"""Convert WBW2 zarr → LeRobot v2.1 for lingbot training (env_type='none').

Action stored in parquet (8-dim, pre-computed relative to episode frame 0):
  [rel_x, rel_y, rel_z, rel_qx, rel_qy, rel_qz, rel_qw, gripper_width_m]

Quaternion convention: xyzw (scipy), with qw at index 6.

Paired config: wan_va/configs/va_umi_wbw_cfg.py
  env_type='none', used_action_channel_ids=[0,1,2,3,4,5,6,28]
  → action_aligned[:, 0:7] = rel EEF, action_aligned[:, 28] = gripper

Run with lingbot-va conda env:
  python convert_wbw2_to_lerobot.py \\
      --zarr  /store/real/jvclark/shared_data/WBW2-addition-wristOvershoot-harderGrasp-ft36_true2.zarr \\
      --out   /store/real/jvclark/lingbot-va/data/umi_wbw_lerobot_v21 \\
      --fps   10 \\
      --task  "erase the drawing"
"""

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
from scipy.spatial.transform import Rotation

TASK_DEFAULT = "erase the drawing"
VIDEO_KEY = "observation.images.camera0"
TARGET_H = 256
TARGET_W = 256

# Resolve ffmpeg: prefer the same env as this interpreter, then PATH
_FFMPEG = shutil.which("ffmpeg", path=os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")) or "ffmpeg"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def episode_slices(episode_ends: np.ndarray):
    starts = np.concatenate([[0], episode_ends[:-1]])
    return [(int(s), int(e)) for s, e in zip(starts, episode_ends)]


_ENCODER_CODEC = {
    "libsvtav1": "av1",
    "libopenh264": "h264",
    "libx264": "h264",
}


def encode_video(frames: np.ndarray, out_path: pathlib.Path, fps: float) -> str:
    """Encode (N, H, W, 3) uint8 frames → mp4 via ffmpeg.

    Tries libsvtav1 first, falls back to libopenh264 then libx264 (CPU).
    Resizes to TARGET_H × TARGET_W via vf scale.
    Returns the codec name that was actually used (e.g. "av1", "h264").
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n, h, w, c = frames.shape
    assert c == 3

    vf = f"scale={TARGET_W}:{TARGET_H}" if (h != TARGET_H or w != TARGET_W) else None

    def _run(encoder: str) -> bool:
        cmd = [
            _FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", str(fps),
            "-i", "pipe:0",
        ]
        if vf:
            cmd += ["-vf", vf]
        if encoder == "libsvtav1":
            cmd += ["-c:v", "libsvtav1", "-crf", "30", "-b:v", "0",
                    "-pix_fmt", "yuv420p"]
        elif encoder == "libopenh264":
            cmd += ["-c:v", "libopenh264", "-b:v", "4M", "-pix_fmt", "yuv420p"]
        else:
            cmd += ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p"]
        cmd += ["-an", str(out_path)]
        r = subprocess.run(cmd, input=frames.tobytes(), capture_output=True)
        if r.returncode != 0:
            sys.stderr.write(r.stderr.decode(errors="replace") + "\n")
        return r.returncode == 0

    for enc in ("libsvtav1", "libopenh264", "libx264"):
        if _run(enc):
            return _ENCODER_CODEC[enc]
    raise RuntimeError(f"ffmpeg failed for {out_path}")


def relative_action(pos: np.ndarray, raa: np.ndarray, grip: np.ndarray) -> np.ndarray:
    """Compute 8-dim relative action [rel_xyz, rel_quat_xyzw, gripper_m].

    Relative to episode frame 0. Quaternion in xyzw (scipy) convention.
    """
    rot = Rotation.from_rotvec(raa.astype(np.float64))
    rel_xyz = (pos - pos[0:1]).astype(np.float32)
    rel_rot = Rotation.from_rotvec(raa[0:1].astype(np.float64)).inv() * rot
    rel_quat_xyzw = rel_rot.as_quat().astype(np.float32)          # (N, 4) xyzw
    return np.concatenate([rel_xyz, rel_quat_xyzw, grip], axis=1)  # (N, 8)


def ep_stats(action: np.ndarray) -> dict:
    """Per-episode action stats for episodes_stats.jsonl."""
    return {
        "min":   action.min(axis=0).tolist(),
        "max":   action.max(axis=0).tolist(),
        "mean":  action.mean(axis=0).tolist(),
        "std":   action.std(axis=0).tolist(),
        "count": [int(len(action))],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", required=True, help="Path to WBW2 .zarr directory")
    ap.add_argument("--out",  required=True, help="Output LeRobot v2.1 dataset root")
    ap.add_argument("--fps",  type=float, default=10.0,
                    help="Recording fps of the zarr data (default: 10)")
    ap.add_argument("--task", default=TASK_DEFAULT, help="Task description string")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    if out.exists() and any(out.iterdir()):
        if not args.overwrite:
            sys.exit(f"Output dir not empty: {out}. Use --overwrite to replace.")
        shutil.rmtree(out)

    (out / "meta").mkdir(parents=True)
    (out / "data" / "chunk-000").mkdir(parents=True)
    (out / "videos" / "chunk-000" / VIDEO_KEY).mkdir(parents=True)

    # Open zarr
    root = zarr.open(args.zarr, "r")
    episode_ends = root["meta"]["episode_ends"][:]
    slices = episode_slices(episode_ends)
    n_episodes = len(slices)
    fps = args.fps
    task = args.task

    print(f"Converting {n_episodes} episodes at {fps} fps → {out}")

    global_index = 0
    episodes_meta = []
    episodes_stats_rows = []
    used_codec = None  # determined from the first encoded video

    for ep_idx, (s, e) in enumerate(slices):
        ep_len = e - s
        if ep_idx % 20 == 0:
            print(f"  episode {ep_idx}/{n_episodes}  (len={ep_len})")

        # --- Read zarr arrays ---
        pos  = np.array(root["data"]["robot0_eef_pos"][s:e],             dtype=np.float32)
        raa  = np.array(root["data"]["robot0_eef_rot_axis_angle"][s:e],  dtype=np.float32)
        grip = np.array(root["data"]["robot0_gripper_width"][s:e],        dtype=np.float32)

        # --- Compute 8-dim relative action ---
        action = relative_action(pos, raa, grip)   # (ep_len, 8)

        # --- Write parquet ---
        frame_idx = np.arange(ep_len, dtype=np.int64)
        timestamps = (frame_idx / fps).astype(np.float32)
        index_col  = np.arange(global_index, global_index + ep_len, dtype=np.int64)

        table = pa.table({
            "timestamp":      pa.array(timestamps),
            "frame_index":    pa.array(frame_idx),
            "episode_index":  pa.array(np.full(ep_len, ep_idx, dtype=np.int64)),
            "index":          pa.array(index_col),
            "task_index":     pa.array(np.zeros(ep_len, dtype=np.int64)),
            "action":            pa.array(action.tolist(), type=pa.list_(pa.float32(), 8)),
            "observation.state": pa.array(grip.tolist(),  type=pa.list_(pa.float32(), 1)),
        })
        pq_path = out / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
        pq.write_table(table, pq_path)

        # --- Encode video ---
        frames = np.array(root["data"]["camera0_rgb"][s:e], dtype=np.uint8)
        vid_path = out / "videos" / "chunk-000" / VIDEO_KEY / f"episode_{ep_idx:06d}.mp4"
        codec = encode_video(frames, vid_path, fps)
        if used_codec is None:
            used_codec = codec

        # --- Accumulate metadata ---
        episodes_meta.append({
            "episode_index": ep_idx,
            "tasks": [task],
            "length": ep_len,
            "action_config": [{
                "start_frame": 0,
                "end_frame": ep_len,
                "action_text": task,
                "skill": "",
            }],
        })
        episodes_stats_rows.append({
            "episode_index": ep_idx,
            "stats": {"action": ep_stats(action)},
        })

        global_index += ep_len

    # --- meta/tasks.jsonl ---
    with open(out / "meta" / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": task}) + "\n")

    # --- meta/episodes.jsonl ---
    with open(out / "meta" / "episodes.jsonl", "w") as f:
        for row in episodes_meta:
            f.write(json.dumps(row) + "\n")

    # --- meta/episodes_stats.jsonl ---
    with open(out / "meta" / "episodes_stats.jsonl", "w") as f:
        for row in episodes_stats_rows:
            f.write(json.dumps(row) + "\n")

    # --- meta/info.json ---
    info = {
        "codebase_version": "v2.1",
        "robot_type": "umi_single_arm",
        "total_episodes": n_episodes,
        "total_frames": int(global_index),
        "total_tasks": 1,
        "total_videos": n_episodes,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{n_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            VIDEO_KEY: {
                "dtype": "video",
                "shape": [3, TARGET_H, TARGET_W],
                "names": ["rgb", "height", "width"],
                "info": {
                    "video.height": TARGET_H,
                    "video.width":  TARGET_W,
                    "video.codec":  used_codec,
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": fps,
                    "video.channels": 3,
                    "has_audio": False,
                },
            },
            "action": {
                "dtype": "float32",
                "shape": [8],
                "names": {
                    "motors": ["rel_x", "rel_y", "rel_z",
                               "rel_qx", "rel_qy", "rel_qz", "rel_qw",
                               "gripper_m"]
                },
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [1],
                "names": {"motors": ["gripper_m"]},
            },
            "timestamp":     {"dtype": "float32", "shape": [1], "names": None},
            "frame_index":   {"dtype": "int64",   "shape": [1], "names": None},
            "episode_index": {"dtype": "int64",   "shape": [1], "names": None},
            "index":         {"dtype": "int64",   "shape": [1], "names": None},
            "task_index":    {"dtype": "int64",   "shape": [1], "names": None},
        },
    }
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    # --- Copy empty_emb.pt ---
    emb_src = pathlib.Path(__file__).parent / "data" / "empty_emb.pt"
    if emb_src.exists():
        shutil.copy(emb_src, out / "empty_emb.pt")
        print(f"Copied empty_emb.pt from {emb_src}")
    else:
        print(f"WARNING: empty_emb.pt not found at {emb_src}. Copy it manually to {out}/empty_emb.pt")

    print(f"\nDone. {n_episodes} episodes, {global_index} frames → {out}")
    print(f"Next: run VAE latent extraction into {out}/latents/")


if __name__ == "__main__":
    main()
