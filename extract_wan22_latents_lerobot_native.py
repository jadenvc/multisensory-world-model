#!/usr/bin/env python3
"""Extract Wan2.2 VAE latents for a LingBot LeRobot v2.1 dataset.

This is the same dataset contract as ``extract_wan22_latents_lerobot.py``, but
uses Diffusers' native ``AutoencoderKLWan.encode`` path instead of LingBot's
streaming wrapper.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLWan
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from tqdm import tqdm
from transformers import T5TokenizerFast, UMT5EncoderModel


DEFAULT_DATASET_ROOT = (
    "/store/real/jvclark/lingbot-va/data/umi_wbw_lerobot_v21/"
    "WBW2-addition-wristOvershoot-harderGrasp-ft36_true2"
)
DEFAULT_CHECKPOINT_ROOT = (
    "/store/real/jvclark/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint-root", default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--camera-key", default="observation.images.camera0")
    parser.add_argument("--target-fps", type=int, default=15)
    parser.add_argument("--ori-fps", type=int, default=60)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--episode-start", type=int, default=None)
    parser.add_argument("--episode-end", type=int, default=None)
    parser.add_argument("--write-empty-emb", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_episodes(dataset_root: Path) -> list[dict]:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    episodes = []
    with episodes_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))
    return episodes


def encode_prompt(
    text_encoder: UMT5EncoderModel,
    tokenizer: T5TokenizerFast,
    prompt: str,
    device: torch.device,
    dtype: torch.dtype,
    max_sequence_length: int,
) -> torch.Tensor:
    prompt = prompt_clean(prompt)
    text_inputs = tokenizer(
        [prompt],
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)
    seq_len = attention_mask.gt(0).sum(dim=1).long()[0].item()

    prompt_embeds = text_encoder(input_ids, attention_mask).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype)[0, :seq_len]
    if prompt_embeds.shape[0] < max_sequence_length:
        pad = prompt_embeds.new_zeros(max_sequence_length - prompt_embeds.shape[0], prompt_embeds.shape[1])
        prompt_embeds = torch.cat([prompt_embeds, pad], dim=0)
    return prompt_embeds.detach().cpu()


def read_sampled_frames(video_path: Path, frame_ids: np.ndarray) -> np.ndarray:
    wanted = set(int(i) for i in frame_ids.tolist())
    frames = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame_index, frame in enumerate(container.decode(stream)):
            if frame_index in wanted:
                frames.append(frame.to_ndarray(format="rgb24"))
                if len(frames) == len(wanted):
                    break
    if len(frames) != len(frame_ids):
        raise RuntimeError(
            f"Decoded {len(frames)} sampled frames from {video_path}, expected {len(frame_ids)}"
        )
    return np.stack(frames, axis=0)


def normalize_latents(vae: AutoencoderKLWan, latents: torch.Tensor) -> torch.Tensor:
    latents_mean = torch.tensor(vae.config.latents_mean, device=latents.device).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std, device=latents.device).view(1, -1, 1, 1, 1)
    return ((latents.float() - latents_mean) / latents_std).to(latents)


@torch.inference_mode()
def encode_video_native(
    vae: AutoencoderKLWan,
    frames: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
    height: int,
    width: int,
) -> torch.Tensor:
    video = torch.from_numpy(frames).float().permute(3, 0, 1, 2)
    if video.shape[-2:] != (height, width):
        video = F.interpolate(video, size=(height, width), mode="bilinear", align_corners=False)
    video = video.unsqueeze(0).to(device=device, dtype=dtype)
    video = video / 255.0 * 2.0 - 1.0

    mu = vae.encode(video).latent_dist.mean
    mu_norm = normalize_latents(vae, mu)
    flat_latent = mu_norm[0].permute(1, 2, 3, 0).reshape(-1, mu_norm.shape[1])
    return flat_latent.detach().cpu()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    checkpoint_root = Path(args.checkpoint_root)
    device = torch.device(args.device)
    dtype = torch_dtype(args.dtype)

    if args.ori_fps % args.target_fps != 0:
        raise ValueError(f"ori_fps={args.ori_fps} must be divisible by target_fps={args.target_fps}")
    frame_stride = args.ori_fps // args.target_fps

    vae = AutoencoderKLWan.from_pretrained(str(checkpoint_root / "vae"), torch_dtype=dtype).to(device)
    vae.eval()

    tokenizer = T5TokenizerFast.from_pretrained(str(checkpoint_root / "tokenizer"))
    text_encoder = UMT5EncoderModel.from_pretrained(
        str(checkpoint_root / "text_encoder"),
        torch_dtype=dtype,
    ).to(device)
    text_encoder.eval()

    empty_emb = encode_prompt(text_encoder, tokenizer, "", device, dtype, args.max_sequence_length)
    if args.write_empty_emb:
        empty_path = dataset_root.parent / "empty_emb.pt"
        torch.save(empty_emb, empty_path)
        print(f"Wrote empty prompt embedding: {empty_path}")

    episodes = load_episodes(dataset_root)
    if args.episode_start is not None:
        episodes = [ep for ep in episodes if ep["episode_index"] >= args.episode_start]
    if args.episode_end is not None:
        episodes = [ep for ep in episodes if ep["episode_index"] <= args.episode_end]
    if args.limit is not None:
        episodes = episodes[: args.limit]

    written = 0
    skipped = 0
    for episode in tqdm(episodes, desc="Extracting latents"):
        episode_index = int(episode["episode_index"])
        chunk = int(episode.get("chunk", episode_index // 1000))
        video_path = (
            dataset_root
            / "videos"
            / f"chunk-{chunk:03d}"
            / args.camera_key
            / f"episode_{episode_index:06d}.mp4"
        )
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        for segment in episode["action_config"]:
            start_frame = int(segment["start_frame"])
            end_frame = int(segment["end_frame"])
            action_text = segment["action_text"]
            latent_dir = dataset_root / "latents" / f"chunk-{chunk:03d}" / args.camera_key
            latent_dir.mkdir(parents=True, exist_ok=True)
            latent_path = latent_dir / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            if latent_path.exists() and not args.overwrite:
                skipped += 1
                continue

            frame_ids = np.arange(start_frame, end_frame, frame_stride, dtype=np.int64)
            if frame_ids.size == 0:
                raise RuntimeError(f"Episode {episode_index} segment {start_frame}:{end_frame} has no sampled frames")

            frames = read_sampled_frames(video_path, frame_ids)
            latent = encode_video_native(vae, frames, device, dtype, args.height, args.width)
            latent_num_frames = (len(frame_ids) - 1) // 4 + 1
            latent_height = args.height // 16
            latent_width = args.width // 16
            expected_rows = latent_num_frames * latent_height * latent_width
            if latent.shape[0] != expected_rows:
                raise RuntimeError(
                    f"{latent_path}: latent rows {latent.shape[0]} != expected {expected_rows}"
                )

            text_emb = encode_prompt(text_encoder, tokenizer, action_text, device, dtype, args.max_sequence_length)
            torch.save(
                {
                    "latent": latent.to(dtype=torch.bfloat16),
                    "latent_num_frames": int(latent_num_frames),
                    "latent_height": int(latent_height),
                    "latent_width": int(latent_width),
                    "video_num_frames": int(len(frame_ids)),
                    "video_height": int(args.height),
                    "video_width": int(args.width),
                    "text_emb": text_emb.to(dtype=torch.bfloat16),
                    "text": action_text,
                    "frame_ids": torch.from_numpy(frame_ids),
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "fps": int(args.target_fps),
                    "ori_fps": int(args.ori_fps),
                },
                latent_path,
            )
            written += 1

    print(f"Done. wrote={written} skipped={skipped} dataset={dataset_root}")


if __name__ == "__main__":
    main()
