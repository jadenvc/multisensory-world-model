"""
Inference speed benchmark for LingBot-VA (robotwin checkpoint).

Run with:
    CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 --master_port=29600 bench_inference.py

Runs two sweeps (model loaded once, _reset() called between configs):
  1. Resolution sweep  — varies image downsampling at fixed denoising steps
  2. Steps sweep       — varies denoising steps at fixed (baseline) resolution

Produces: bench_results.png  (two-panel figure)
"""
# Stub out flash_attn before any other imports — it is imported unconditionally
# by both wan_va and diffusers, but the installed version was compiled against a
# different PyTorch and causes a symbol mismatch.  The model uses attn_mode='torch'
# (PyTorch SDPA) so flash_attn is never actually called.
import sys, types as _types, importlib.machinery as _imach
for _mod in ('flash_attn_2_cuda', 'flash_attn', 'flash_attn.flash_attn_interface',
             'flash_attn_interface'):
    _m = _types.ModuleType(_mod)
    _m.__spec__ = _imach.ModuleSpec(_mod, loader=None)
    _m.__path__ = []  # mark as package so sub-imports don't break
    _m.flash_attn_func = None
    _m.flash_attn_varlen_func = None
    sys.modules[_mod] = _m

import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "wan_va"))

from configs import VA_CONFIGS
from distributed.util import init_distributed
from utils import init_logger
import modules.utils as _mutils
from modules.model import WanTransformer3DModel

# Force attn_mode='torch' (PyTorch SDPA) — avoids flash_attn at runtime
_orig_load_transformer = _mutils.load_transformer
def _load_transformer_torch(path, torch_dtype, torch_device):
    model = WanTransformer3DModel.from_pretrained(path, torch_dtype=torch_dtype, attn_mode='torch')
    return model.to(torch_device)
_mutils.load_transformer = _load_transformer_torch

from wan_va_server import VA_Server

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = os.path.join(os.path.dirname(__file__),
                            "checkpoints/lingbot-va-posttrain-robotwin")
CONFIG_NAME  = "robotwin"
NUM_WARMUP   = 1
NUM_TIMED    = 102
PROMPT       = "Pick the object and place it in the box"
SAVE_ROOT    = "/tmp/lingbot_bench"
PLOT_DIR     = os.path.dirname(__file__)

# ── Which sweeps to run ───────────────────────────────────────────────────────
# Options: "resolution", "steps", "both"
SWEEP = "resolution"

# ── Sweep 1: resolutions (fixed steps) ───────────────────────────────────────
# Constraints for robotwin_tshape:
#   height must be divisible by 64  (so latent_h = (h//16)*3//2 is even)
#   width  must be divisible by 32  (so latent_w = w//16 is even for patch_size[2]=2)
# Listed high→low; comment out entries to skip.
# WARNING: resolutions above 512×640 will be very slow (OOM risk at 4K).
RESOLUTIONS = [
    # ("2048×2560",  2048, 2560),  # ~4K (4:5 ratio)  — 7680 tokens/frame
    # ("1024×1280",  1024, 1280),  # 2K               — 1920 tokens/frame
    # (" 512× 640",   512,  640),  # 2× baseline      —  480 tokens/frame
    (" 384× 480",   384,  480),  # 1.5× baseline    —  270 tokens/frame
    (" 256× 320",   256,  320),  # baseline         —  120 tokens/frame
    (" 192× 320",   192,  320),  # ¾ height         —   90 tokens/frame
    (" 128× 160",   128,  160),  # ½ both           —   30 tokens/frame
]

# ── Sweep 2: denoising steps (fixed baseline resolution 256×320) ─────────────
# Each entry: (label, video_steps, action_steps)
STEP_CONFIGS = [
    ("1v + 10a", 1, 10),   # baseline
    ("2v + 10a", 2, 10),
    (" 4v + 10a",  4, 10),
    ("8v +  10a", 8,  10),
]
# ─────────────────────────────────────────────────────────────────────────────

init_logger()
os.makedirs(SAVE_ROOT, exist_ok=True)

rank       = int(os.getenv("RANK", 0))
local_rank = int(os.getenv("LOCAL_RANK", 0))
world_size = int(os.getenv("WORLD_SIZE", 1))
init_distributed(world_size, local_rank, rank)

config = VA_CONFIGS[CONFIG_NAME]
config.wan22_pretrained_model_name_or_path = MODEL_PATH
config.local_rank  = local_rank
config.rank        = rank
config.world_size  = world_size
config.save_root   = SAVE_ROOT
config.frame_chunk_size = 4
config.attn_window = config.attn_window * 2  # 72 → 216  (~3× KV cache capacity)

# ── Load model once ───────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Config     : {CONFIG_NAME}")
print(f"  Model path : {MODEL_PATH}")
print(f"  Device     : cuda:{local_rank}  (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','unset')})")
print(f"  Denoising  : {config.num_inference_steps} video + {config.action_num_inference_steps} action steps")
print(f"  Chunk size : {config.frame_chunk_size} frames")
print(f"{'='*60}\n")

print("Loading model...")
t_load = time.perf_counter()
model = VA_Server(config)
torch.cuda.synchronize()
print(f"Model loaded in {time.perf_counter() - t_load:.1f}s\n")

def run_benchmark(label, dummy_obs):
    """Run NUM_WARMUP + NUM_TIMED chunks, return (mean_ms, std_ms, actions_per_sec, all_times_ms).
    all_times_ms includes every call (warmup + timed) so the per-call latency plot captures
    the cache-filling ramp-up from call 0.
    """
    times = []
    all_times = []
    frame_st_id = 0
    for i in range(NUM_WARMUP + NUM_TIMED):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        action, _ = model._infer(dummy_obs, frame_st_id=frame_st_id)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        tag = "WARMUP" if i < NUM_WARMUP else "TIMED "
        print(f"    [{tag}] chunk {i:2d}  |  {elapsed_ms:7.1f} ms")
        all_times.append(elapsed_ms)
        if i >= NUM_WARMUP:
            times.append(elapsed_ms)
        frame_st_id += config.frame_chunk_size
    times_t = torch.tensor(times)
    mean_ms = times_t.mean().item()
    std_ms  = times_t.std().item()
    aps     = (config.frame_chunk_size * config.action_per_frame) / (mean_ms / 1000)
    print(f"  → Mean {mean_ms:.1f} ms  ±  {std_ms:.1f} ms  →  {aps:.1f} actions/sec")
    return mean_ms, std_ms, aps, all_times


BASE_H, BASE_W        = config.height, config.width
BASE_VIDEO_STEPS      = 1
BASE_ACTION_STEPS     = 10
res_results           = []
step_results          = []

# ── Sweep 1: resolutions ──────────────────────────────────────────────────────
if SWEEP in ("resolution", "both"):
    print(f"\n{'='*60}")
    print(f"  SWEEP 1: resolutions  "
          f"({BASE_VIDEO_STEPS}v + {BASE_ACTION_STEPS}a steps)")
    print(f"{'='*60}")

    for label, H, W in RESOLUTIONS:
        config.num_inference_steps        = BASE_VIDEO_STEPS
        config.action_num_inference_steps = BASE_ACTION_STEPS
        latent_h = ((H // 16) * 3) // 2
        latent_w = W // 16
        patch_t, patch_h, patch_w = config.patch_size
        tokens_per_frame = (latent_h * latent_w) // (patch_h * patch_w)

        print(f"\n  {label}  →  {tokens_per_frame} tokens/frame")
        config.height, config.width = H, W
        model._reset(prompt=PROMPT)
        torch.cuda.synchronize()

        dummy_obs = {"obs": [
            {k: np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
             for k in config.obs_cam_keys}
        ]}
        full_label = f"{label.strip()} {BASE_VIDEO_STEPS}v_{BASE_ACTION_STEPS}a"
        mean_ms, std_ms, aps, all_times = run_benchmark(full_label, dummy_obs)
        res_results.append((full_label, tokens_per_frame, mean_ms, std_ms, aps, all_times))

    # Restore baseline resolution for sweep 2
    config.height, config.width = BASE_H, BASE_W

# ── Sweep 2: denoising steps ──────────────────────────────────────────────────
if SWEEP in ("steps", "both"):
    print(f"\n{'='*60}")
    print(f"  SWEEP 2: denoising steps  ({BASE_H}×{BASE_W})")
    print(f"{'='*60}")

    dummy_obs_base = {"obs": [
        {k: np.random.randint(0, 256, (BASE_H, BASE_W, 3), dtype=np.uint8)
         for k in config.obs_cam_keys}
    ]}

    for label, vsteps, asteps in STEP_CONFIGS:
        print(f"\n  {label}")
        config.num_inference_steps        = vsteps
        config.action_num_inference_steps = asteps
        model._reset(prompt=PROMPT)
        torch.cuda.synchronize()
        mean_ms, std_ms, aps, all_times = run_benchmark(label, dummy_obs_base)
        step_results.append((label, vsteps, asteps, mean_ms, std_ms, aps, all_times))

    # Restore defaults
    config.num_inference_steps        = BASE_VIDEO_STEPS
    config.action_num_inference_steps = BASE_ACTION_STEPS

# ── Print summary tables ──────────────────────────────────────────────────────
if res_results:
    print(f"\n{'='*60}  SWEEP 1: resolutions")
    print(f"  {'Label':<14}  {'Tokens/frame':>13}  {'Mean ms':>8}  {'Std ms':>7}  {'Act/s':>7}")
    print(f"  {'─'*14}  {'─'*13}  {'─'*8}  {'─'*7}  {'─'*7}")
    for label, tpf, mean_ms, std_ms, aps, *_ in res_results:
        print(f"  {label:<14}  {tpf:>13d}  {mean_ms:>8.1f}  {std_ms:>7.1f}  {aps:>7.1f}")

if step_results:
    print(f"\n{'='*60}  SWEEP 2: denoising steps")
    print(f"  {'Label':<12}  {'Video steps':>11}  {'Action steps':>12}  {'Mean ms':>8}  {'Std ms':>7}  {'Act/s':>7}")
    print(f"  {'─'*12}  {'─'*11}  {'─'*12}  {'─'*8}  {'─'*7}  {'─'*7}")
    for label, vsteps, asteps, mean_ms, std_ms, aps, *_ in step_results:
        print(f"  {label:<12}  {vsteps:>11d}  {asteps:>12d}  {mean_ms:>8.1f}  {std_ms:>7.1f}  {aps:>7.1f}")
print()

# ── Build plot filename ───────────────────────────────────────────────────────
_parts = [f"n{NUM_TIMED}", f"chunk{config.frame_chunk_size}", f"{config.action_per_frame}apf"]
if SWEEP in ("resolution", "both"):
    _hs = [H for _, H, W in RESOLUTIONS]
    _ws = [W for _, H, W in RESOLUTIONS]
    _parts.append(f"res{len(RESOLUTIONS)}_{min(_hs)}x{min(_ws)}-{max(_hs)}x{max(_ws)}")
if SWEEP in ("steps", "both"):
    _vs = sorted({v for _, v, _ in STEP_CONFIGS})
    _as = sorted({a for _, _, a in STEP_CONFIGS})
    _v_str = f"{min(_vs)}-{max(_vs)}v" if len(_vs) > 1 else f"{_vs[0]}v"
    _a_str = f"{min(_as)}-{max(_as)}a" if len(_as) > 1 else f"{_as[0]}a"
    _parts.append(f"steps_{_v_str}_{_a_str}")
PLOT_PATH = os.path.join(PLOT_DIR, f"bench_results_{'_'.join(_parts)}.png")
print(f"Plot will be saved to: {PLOT_PATH}\n")

# ── Plot ──────────────────────────────────────────────────────────────────────
chunk_str = (f"chunk={config.frame_chunk_size} frames, "
             f"{config.action_per_frame} act/frame")

def _plot_sweep(ax_ms, ax_aps, xs, results, color, xlabel, title_base, invert_x=False):
    """Plot latency and actions/sec side-by-side for one sweep."""
    labels  = [r[0] for r in results]
    mean_ms = [r[-4] for r in results]
    std_ms  = [r[-3] for r in results]
    aps     = [r[-2] for r in results]
    aps_err = [s / m * a for s, m, a in zip(std_ms, mean_ms, aps)]

    for ax, ys, yerrs, ylabel in [
        (ax_ms,  mean_ms, std_ms,  "Latency (ms)"),
        (ax_aps, aps,     aps_err, "Actions / second"),
    ]:
        ax.errorbar(xs, ys, yerr=yerrs,
                    fmt='o-', linewidth=2, markersize=8, capsize=5, color=color)
        for x, y, lbl in zip(xs, ys, labels):
            ax.annotate(lbl.strip(), xy=(x, y),
                        xytext=(6, 4), textcoords='offset points', fontsize=9)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3)
        if invert_x:
            ax.invert_xaxis()

    ax_ms.set_title(f"{title_base}  —  latency\n[{chunk_str}]", fontsize=10)
    ax_aps.set_title(f"{title_base}  —  actions/sec\n[{chunk_str}]", fontsize=10)

n_sweeps = (1 if res_results else 0) + (1 if step_results else 0)
fig, axes = plt.subplots(1, 2 * n_sweeps,
                         figsize=(7 * 2 * n_sweeps, 4),
                         squeeze=False)
axes = axes[0]   # shape: (2 * n_sweeps,)
ax_iter = iter(axes)

if res_results:
    ax_ms, ax_aps = next(ax_iter), next(ax_iter)
    _plot_sweep(ax_ms, ax_aps,
                [r[1] for r in res_results],
                res_results,
                color='steelblue',
                xlabel="Latent tokens / frame",
                title_base=f"Sweep 1: image resolution\n({BASE_VIDEO_STEPS}v + {BASE_ACTION_STEPS}a steps)",
                invert_x=True)   # high res (many tokens) on left, low res on right

if step_results:
    ax_ms, ax_aps = next(ax_iter), next(ax_iter)
    _plot_sweep(ax_ms, ax_aps,
                [r[2] for r in step_results],   # action steps on x-axis (dominant)
                step_results,
                color='darkorange',
                xlabel="Action denoising steps",
                title_base=f"Sweep 2: denoising steps\n({BASE_H}×{BASE_W})")

fig.tight_layout()
fig.savefig(PLOT_PATH, dpi=150)
print(f"Plot saved to {PLOT_PATH}")

# ── Per-call latency plot ─────────────────────────────────────────────────────
# One line per config, x = call index (0 = first warmup call), y = latency ms.
# Warmup calls are shown with a dashed region so the cache-filling ramp is visible.
all_per_call = []  # (label, color, [ms per call])
if res_results:
    cmap = plt.cm.Blues
    n = len(res_results)
    for i, row in enumerate(res_results):
        label, _tpf = row[0], row[1]
        all_times = row[-1]
        all_per_call.append((label.strip(), cmap(0.4 + 0.6 * i / max(n - 1, 1)), all_times))
if step_results:
    cmap = plt.cm.Oranges
    n = len(step_results)
    for i, row in enumerate(step_results):
        label = row[0]
        all_times = row[-1]
        all_per_call.append((label.strip(), cmap(0.4 + 0.6 * i / max(n - 1, 1)), all_times))

if all_per_call:
    fig2, ax = plt.subplots(figsize=(10, 4))
    for label, color, times in all_per_call:
        xs = list(range(len(times)))
        # shade warmup calls differently
        ax.plot(xs[:NUM_WARMUP], times[:NUM_WARMUP],
                linestyle='--', marker='o', markersize=5, color=color, alpha=0.5)
        ax.plot(xs[NUM_WARMUP - 1:], times[NUM_WARMUP - 1:],
                linestyle='-', marker='o', markersize=5, color=color, label=label)
    # mark where the cache is full
    cache_fill_call = (72 // config.frame_chunk_size)  # attn_window / chunk_size
    ax.axvline(cache_fill_call, color='gray', linestyle=':', linewidth=1.5,
               label=f'cache full (~call {cache_fill_call})')
    ax.axvspan(0, NUM_WARMUP - 0.5, color='gray', alpha=0.07, label=f'warmup ({NUM_WARMUP} call(s))')
    ax.set_xlabel("Inference call index", fontsize=11)
    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.set_title(f"Per-call latency  —  [{chunk_str}]\n"
                 f"(dashed = warmup, dotted line = cache full at ~{cache_fill_call} calls)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    PERCALL_PATH = PLOT_PATH.replace(".png", "_percall.png")
    fig2.savefig(PERCALL_PATH, dpi=150)
    print(f"Per-call plot saved to {PERCALL_PATH}")