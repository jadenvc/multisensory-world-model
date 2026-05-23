#!/usr/bin/env python3
"""Extract LingBot-VA Wan2.2 latents for a LeRobot v2.1 dataset.

The output follows the README contract:

    latents/chunk-000/<camera_key>/episode_000000_0_<length>.pth

Each file contains flattened VAE latents, Wan text embeddings, sampled frame
ids, and the segment metadata from ``meta/episodes.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLWan
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from transformers import T5TokenizerFast, UMT5EncoderModel
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
sys.path.append(str(REPO_ROOT))


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


class WanVAEStreamingWrapper:
    def __init__(self, vae_model):
        self.vae = vae_model
        self.encoder = vae_model.encoder
        self.quant_conv = vae_model.quant_conv

        if hasattr(self.vae, "_cached_conv_counts"):
            self.enc_conv_num = self.vae._cached_conv_counts["encoder"]
        else:
            self.enc_conv_num = sum(
                1 for module in self.encoder.modules() if module.__class__.__name__ == "WanCausalConv3d"
            )
        self.clear_cache()

    def clear_cache(self) -> None:
        self.feat_cache = [None] * self.enc_conv_num

    def encode_chunk(self, x_chunk: torch.Tensor) -> torch.Tensor:
        if hasattr(self.vae.config, "patch_size") and self.vae.config.patch_size is not None:
            x_chunk = patchify(x_chunk, self.vae.config.patch_size)
        feat_idx = [0]
        out = self.encoder(x_chunk, feat_cache=self.feat_cache, feat_idx=feat_idx)
        return self.quant_conv(out)


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    if patch_size is None or patch_size == 1:
        return x
    batch_size, channels, frames, height, width = x.shape
    x = x.view(batch_size, channels, frames, height // patch_size, patch_size, width // patch_size, patch_size)
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    return x.view(batch_size, channels * patch_size * patch_size, frames, height // patch_size, width // patch_size)


def load_vae(vae_path: Path, dtype: torch.dtype, device: torch.device):
    vae = AutoencoderKLWan.from_pretrained(str(vae_path), torch_dtype=dtype)
    return vae.to(device)


def load_tokenizer(tokenizer_path: Path):
    return T5TokenizerFast.from_pretrained(str(tokenizer_path))


def load_text_encoder(text_encoder_path: Path, dtype: torch.dtype, device: torch.device):
    text_encoder = UMT5EncoderModel.from_pretrained(str(text_encoder_path), torch_dtype=dtype)
    return text_encoder.to(device)


def load_episodes(dataset_root: Path) -> list[dict]:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    episodes = []
    with episodes_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))
    return episodes


def encode_prompt(
    text_encoder,
    tokenizer,
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
    prompt_embeds = prompt_embeds.to(dtype=dtype)[0]
    prompt_embeds = prompt_embeds[:seq_len]
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


def normalize_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    latents_mean = torch.tensor(vae.config.latents_mean, device=latents.device).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std, device=latents.device).view(1, -1, 1, 1, 1)
    return ((latents.float() - latents_mean) / latents_std).to(latents)


@torch.inference_mode()
def encode_video(
    vae,
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

    # vae.encode() uses the correct causal chunking: 1 frame first, then 4 at a time.
    # The streaming wrapper (encode_chunk) is for inference only and breaks on long videos
    # because on the first call the temporal downsampling is skipped (cache=None), causing
    # a shape mismatch at the residual connection.
    mu = vae.encode(video).latent_dist.mean  # (1, C, F, H//8, W//8)
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

    vae = load_vae(checkpoint_root / "vae", dtype=dtype, device=device)
    vae.eval()
    streaming_vae = WanVAEStreamingWrapper(vae)

    tokenizer = load_tokenizer(checkpoint_root / "tokenizer")
    text_encoder = load_text_encoder(checkpoint_root / "text_encoder", dtype=dtype, device=device)
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
        episode_index = episode["episode_index"]
        chunk = episode.get("chunk", episode_index // 1000)
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
            latent = encode_video(vae, frames, device, dtype, args.height, args.width)
            latent_num_frames = (len(frame_ids) - 1) // 4 + 1
            latent_height = args.height // 16
            latent_width = args.width // 16
            expected_rows = latent_num_frames * latent_height * latent_width
            if latent.shape[0] != expected_rows:
                raise RuntimeError(
                    f"{latent_path}: latent rows {latent.shape[0]} != expected {expected_rows}"
                )
            text_emb = encode_prompt(text_encoder, tokenizer, action_text, device, dtype, args.max_sequence_length)
            payload = {
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
            }
            torch.save(payload, latent_path)
            written += 1

    print(f"Done. wrote={written} skipped={skipped} dataset={dataset_root}")


if __name__ == "__main__":
    main()
