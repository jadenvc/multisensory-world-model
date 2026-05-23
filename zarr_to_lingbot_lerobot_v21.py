#!/usr/bin/env python3
"""Convert a UMI zarr dataset to the LeRobot v2.1 layout used by LingBot-VA.

This follows the LingBot-VA README post-training data contract:

1. Raw zarr -> LeRobot dataset with videos, action, metadata.
2. Each episode has an ``action_config`` segment in ``meta/episodes.jsonl``.
3. Wan2.2 VAE latents can later be written under ``latents/`` with matching
   ``episode_{episode_index}_{start_frame}_{end_frame}.pth`` names.

The action written here is a compact single-arm absolute EEF action:
``[x, y, z, qx, qy, qz, qw, gripper_width]``.
The LingBot dataset loader maps it to relative pose when ``env_type='umi_single'``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import zarr
from scipy.spatial.transform import Rotation


DEFAULT_ZARR = (
    "/local/real/jvclark/unified_video_action/uva/umi_data/umi_ft/"
    "WBW2-addition-wristOvershoot-harderGrasp-ft36_true2.zarr"
)
DEFAULT_OUTPUT = (
    "/store/real/jvclark/lingbot-va/data/umi_wbw_lerobot_v21/"
    "WBW2-addition-wristOvershoot-harderGrasp-ft36_true2"
)
DEFAULT_EMPTY_EMB = "/store/real/jvclark/lingbot-va/data/empty_emb.pt"


@dataclass
class OnlineStats:
    count: int = 0
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None
    min: np.ndarray | None = None
    max: np.ndarray | None = None

    def update(self, values: np.ndarray) -> None:
        arr = np.asarray(values)
        if arr.ndim == 1:
            arr = arr[:, None]
        arr = arr.reshape(arr.shape[0], -1).astype(np.float64)
        if arr.shape[0] == 0:
            return

        batch_count = arr.shape[0]
        batch_mean = arr.mean(axis=0)
        batch_m2 = ((arr - batch_mean) ** 2).sum(axis=0)
        batch_min = arr.min(axis=0)
        batch_max = arr.max(axis=0)

        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            self.min = batch_min
            self.max = batch_max
            return

        assert self.mean is not None
        assert self.m2 is not None
        assert self.min is not None
        assert self.max is not None

        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * (batch_count / total)
        self.m2 = (
            self.m2
            + batch_m2
            + (delta**2) * (self.count * batch_count / total)
        )
        self.min = np.minimum(self.min, batch_min)
        self.max = np.maximum(self.max, batch_max)
        self.count = total

    def as_lerobot(self) -> dict[str, list[Any]]:
        if self.count == 0:
            raise ValueError("Cannot finalize empty stats")
        assert self.mean is not None
        assert self.m2 is not None
        assert self.min is not None
        assert self.max is not None
        std = np.sqrt(self.m2 / self.count)
        return {
            "min": self.min.tolist(),
            "max": self.max.tolist(),
            "mean": self.mean.tolist(),
            "std": std.tolist(),
            "count": [int(self.count)],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr", default=DEFAULT_ZARR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--repo-id", default="local/umi_wbw_lerobot_v21")
    parser.add_argument("--task", default="erase the drawing")
    parser.add_argument("--camera-key", default="observation.images.camera0")
    parser.add_argument("--source-fps", type=float, default=60.0)
    parser.add_argument("--video-height", type=int, default=256)
    parser.add_argument("--video-width", type=int, default=256)
    parser.add_argument("--encoder", default="h264_nvenc")
    parser.add_argument("--ffmpeg", default=os.environ.get("FFMPEG", "ffmpeg"))
    parser.add_argument("--skip-source-episodes", default="14")
    parser.add_argument("--empty-emb", default=DEFAULT_EMPTY_EMB)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_episode_list(value: str) -> set[int]:
    if value.strip() == "":
        return set()
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def episode_slices(episode_ends: np.ndarray) -> list[slice]:
    starts = np.concatenate([[0], episode_ends[:-1]])
    return [slice(int(start), int(end)) for start, end in zip(starts, episode_ends)]


def get_episode_chunk(episode_index: int, chunk_size: int = 1000) -> int:
    return episode_index // chunk_size


def action_from_zarr(data_group: zarr.Group, sl: slice) -> np.ndarray:
    pos = np.asarray(data_group["robot0_eef_pos"][sl], dtype=np.float32)
    rotvec = np.asarray(data_group["robot0_eef_rot_axis_angle"][sl], dtype=np.float32)
    gripper = np.asarray(data_group["robot0_gripper_width"][sl], dtype=np.float32)
    quat_xyzw = Rotation.from_rotvec(rotvec).as_quat().astype(np.float32)
    if gripper.ndim == 1:
        gripper = gripper[:, None]
    return np.concatenate([pos, quat_xyzw, gripper], axis=1).astype(np.float32)


def encode_video(
    frames: np.ndarray,
    output_path: pathlib.Path,
    ffmpeg: str,
    fps: float,
    height: int,
    width: int,
    encoder: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n, h, w, c = frames.shape
    if c != 3:
        raise ValueError(f"Expected RGB frames, got shape {frames.shape}")

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-vf",
        f"scale={width}:{height}",
        "-c:v",
        encoder,
    ]
    if encoder.endswith("_nvenc"):
        cmd += ["-preset", "p5", "-cq", "30", "-b:v", "0"]
    else:
        cmd += ["-crf", "20", "-preset", "veryfast"]
    cmd += ["-pix_fmt", "yuv420p", "-an", str(output_path)]

    result = subprocess.run(cmd, input=frames.tobytes(), capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg failed for {output_path}: {stderr}")


def stats_for_episode(
    episode_index: int,
    global_start: int,
    length: int,
    action: np.ndarray,
    state: np.ndarray,
    timestamp: np.ndarray,
) -> dict[str, Any]:
    frame_index = np.arange(length, dtype=np.int64)
    global_index = np.arange(global_start, global_start + length, dtype=np.int64)
    constants = {
        "episode_index": np.full((length, 1), episode_index, dtype=np.int64),
        "index": global_index[:, None],
        "frame_index": frame_index[:, None],
        "task_index": np.zeros((length, 1), dtype=np.int64),
        "timestamp": timestamp[:, None],
        "action": action,
        "observation.state": state,
    }
    out = {}
    for key, values in constants.items():
        stat = OnlineStats()
        stat.update(values)
        out[key] = stat.as_lerobot()
    return out


def build_info(
    total_episodes: int,
    total_frames: int,
    camera_key: str,
    task_count: int,
    fps: float,
    video_height: int,
    video_width: int,
    repo_id: str,
) -> dict[str, Any]:
    return {
        "codebase_version": "v2.1",
        "robot_type": "umi",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": task_count,
        "total_videos": total_episodes,
        "total_chunks": max(1, math.ceil(total_episodes / 1000)),
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            camera_key: {
                "dtype": "video",
                "shape": [3, video_height, video_width],
                "names": ["rgb", "height", "width"],
                "info": {
                    "video.height": video_height,
                    "video.width": video_width,
                    "video.codec": "h264",
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
                    "motors": ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper_width"]
                },
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [8],
                "names": {
                    "motors": ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper_width"]
                },
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
        "repo_id": repo_id,
    }


def main() -> None:
    args = parse_args()
    zarr_path = pathlib.Path(args.zarr)
    output = pathlib.Path(args.output)
    skip_source_episodes = parse_episode_list(args.skip_source_episodes)

    root = zarr.open(str(zarr_path), mode="r")
    data = root["data"]
    episode_ends = np.asarray(root["meta"]["episode_ends"])
    slices = episode_slices(episode_ends)
    selected = [
        (source_ep, sl)
        for source_ep, sl in enumerate(slices)
        if source_ep not in skip_source_episodes
    ]

    print(f"Input zarr: {zarr_path}")
    print(f"Output: {output}")
    print(f"Source episodes: {len(slices)}")
    print(f"Selected episodes: {len(selected)}")
    print(f"Skipped source episodes: {sorted(skip_source_episodes)}")
    if args.dry_run:
        return

    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output is not empty: {output}. Pass --overwrite.")
        shutil.rmtree(output)

    (output / "meta").mkdir(parents=True, exist_ok=True)
    (output / "data").mkdir(parents=True, exist_ok=True)
    (output / "videos").mkdir(parents=True, exist_ok=True)

    dataset_root = output.parent
    empty_emb = pathlib.Path(args.empty_emb)
    if empty_emb.exists():
        shutil.copy2(empty_emb, dataset_root / "empty_emb.pt")
    else:
        print(f"Warning: empty_emb not found at {empty_emb}")

    global_index = 0
    episodes_rows = []
    episode_stats_rows = []
    total_stats: dict[str, OnlineStats] = {}

    for episode_index, (source_episode_index, sl) in enumerate(selected):
        length = int(sl.stop - sl.start)
        chunk = get_episode_chunk(episode_index)
        data_dir = output / "data" / f"chunk-{chunk:03d}"
        video_dir = output / "videos" / f"chunk-{chunk:03d}" / args.camera_key
        data_dir.mkdir(parents=True, exist_ok=True)
        video_dir.mkdir(parents=True, exist_ok=True)

        action = action_from_zarr(data, sl)
        state = action.copy()
        frame_index = np.arange(length, dtype=np.int64)
        timestamp = frame_index.astype(np.float32) / float(args.source_fps)
        index = np.arange(global_index, global_index + length, dtype=np.int64)

        df = pd.DataFrame(
            {
                "timestamp": timestamp,
                "frame_index": frame_index,
                "episode_index": np.full(length, episode_index, dtype=np.int64),
                "index": index,
                "task_index": np.zeros(length, dtype=np.int64),
                "action": [row for row in action],
                "observation.state": [row for row in state],
            }
        )
        df.to_parquet(data_dir / f"episode_{episode_index:06d}.parquet", engine="pyarrow")

        frames = np.asarray(data["camera0_rgb"][sl], dtype=np.uint8)
        encode_video(
            frames=frames,
            output_path=video_dir / f"episode_{episode_index:06d}.mp4",
            ffmpeg=args.ffmpeg,
            fps=args.source_fps,
            height=args.video_height,
            width=args.video_width,
            encoder=args.encoder,
        )

        episodes_rows.append(
            {
                "episode_index": episode_index,
                "tasks": [args.task],
                "length": length,
                "action_config": [
                    {
                        "start_frame": 0,
                        "end_frame": length,
                        "action_text": args.task,
                        "skill": "",
                    }
                ],
            }
        )
        episode_stats_rows.append(
            {
                "episode_index": episode_index,
                "stats": stats_for_episode(
                    episode_index, global_index, length, action, state, timestamp
                ),
            }
        )

        for key, values in {
            "action": action,
            "observation.state": state,
            "timestamp": timestamp[:, None],
        }.items():
            total_stats.setdefault(key, OnlineStats()).update(values)

        global_index += length
        print(
            f"episode {episode_index:06d} from source {source_episode_index:06d}: "
            f"{length} frames"
        )

    with open(output / "meta" / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": args.task}) + "\n")

    with open(output / "meta" / "episodes.jsonl", "w") as f:
        for row in episodes_rows:
            f.write(json.dumps(row) + "\n")

    with open(output / "meta" / "episodes_stats.jsonl", "w") as f:
        for row in episode_stats_rows:
            f.write(json.dumps(row) + "\n")

    info = build_info(
        total_episodes=len(episodes_rows),
        total_frames=global_index,
        camera_key=args.camera_key,
        task_count=1,
        fps=args.source_fps,
        video_height=args.video_height,
        video_width=args.video_width,
        repo_id=args.repo_id,
    )
    with open(output / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    with open(output / "meta" / "stats.json", "w") as f:
        json.dump({key: stat.as_lerobot() for key, stat in total_stats.items()}, f, indent=2)

    print(f"Converted {len(episodes_rows)} episodes / {global_index} frames")
    print(f"LeRobot dataset: {output}")
    print(f"Copied empty embedding to: {dataset_root / 'empty_emb.pt'}")


if __name__ == "__main__":
    main()
