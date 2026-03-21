# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import gc
import json
import logging
import math
import os
import random
import sys
import time
import warnings
from contextlib import contextmanager
from datetime import datetime

warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.modules.model import sinusoidal_embedding_1d
from wan.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                                  get_sampling_sigmas, retrieve_timesteps)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_image, cache_video, str2bool


# =============================================================================
# SenCache: Jacobian-based sensitivity caching for diffusion model inference
# =============================================================================

def sencache_forward(
    self,
    x,
    t,
    context,
    seq_len,
    clip_fea=None,
    y=None,
):
    """
    Forward pass with SenCache (Jacobian-based sensitivity caching).

    Uses a dual threshold strategy and a maximum consecutive skip limit (K).
    The caching decision is made on the CONDITIONAL pass (even cnt) only.
    If it's safe to skip, both conditional and unconditional passes are skipped.
    """
    if self.model_type == 'i2v':
        assert clip_fea is not None and y is not None

    # --- Device setup ---
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # --- SenCache: decide whether to skip this step ---
    z_t = x[0]  # VAE latent
    t_val = t
    is_cond = (self.cnt % 2 == 0)  # even = conditional, odd = unconditional
    should_calculate = True

    # Dual threshold: early steps use a tighter threshold
    if self.cnt < self.threshold_switch_step:
        current_threshold = self.sencache_threshold_start
    else:
        current_threshold = self.sencache_threshold_main

    is_in_skip_zone = (self.cnt >= self.ret_steps) and (self.cnt < self.cutoff_steps)

    if is_cond:
        # CONDITIONAL pass: make the caching decision for this denoising step
        if is_in_skip_zone and self.cached_z is not None:
            norm_delta_z = torch.norm(z_t - self.cached_z).item()
            norm_delta_t = torch.abs(t_val - self.cached_t).item()

            sensitivity_error = (self.cached_J_z_norm * norm_delta_z) + (self.cached_J_t_norm * norm_delta_t)

            if sensitivity_error < current_threshold and self.accumulated_skips < self.sencache_K:
                should_calculate = False
                self.skip_this_step = True
                self.accumulated_skips += 1
                self.current_skip_count += 2  # skipping both cond and uncond
            else:
                self.skip_this_step = False
                self.accumulated_skips = 0
        else:
            self.skip_this_step = False
            self.accumulated_skips = 0
    else:
        # UNCONDITIONAL pass: follow the decision made by the conditional pass
        should_calculate = not self.skip_this_step

    # --- Standard model embeddings ---
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                   dim=1) for u in x
    ])

    # Time embeddings
    with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # Context embeddings
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat(
                [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    if clip_fea is not None:
        context_clip = self.img_emb(clip_fea)
        context = torch.concat([context_clip, context], dim=1)

    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens)

    # --- Main computation: skip or calculate ---
    if not should_calculate:
        # SKIP: apply cached residual
        cached_residual = self.cached_residual_cond if is_cond else self.cached_residual_uncond
        x = x + cached_residual
    else:
        # CALCULATE: run transformer blocks and update cache
        ori_x_patched = x.clone()

        for block in self.blocks:
            x = block(x, **kwargs)

        new_residual = x - ori_x_patched

        # Store residual (separate for cond/uncond since outputs differ)
        if is_cond:
            self.cached_residual_cond = new_residual.detach()
            # Update z/t reference point and refresh J norms for this timestep
            self.cached_z = z_t.detach()
            self.cached_t = t_val.detach()
            current_t = t_val.item()
            nearest_index = np.argmin(np.abs(self.sencache_timesteps_array - current_t))
            self.cached_J_z_norm = self.J_z_norm[nearest_index]
            self.cached_J_t_norm = self.J_t_norm[nearest_index]
        else:
            self.cached_residual_uncond = new_residual.detach()

    # --- Post-processing & step counter ---
    x = self.head(x, e)
    x = self.unpatchify(x, grid_sizes)

    self.cnt += 1
    if self.cnt >= self.num_steps:
        # Log skip count for this generation
        self.all_skip_counts.append(self.current_skip_count)
        total_calls = self.num_steps
        logging.info(
            f"[SenCache] Generation complete: "
            f"skipped {self.current_skip_count}/{total_calls} calls "
            f"({100 * self.current_skip_count / total_calls:.1f}%)"
        )
        # Reset all state for next generation
        self.current_skip_count = 0
        self.cnt = 0
        self.cached_z = None
        self.cached_t = None
        self.cached_residual_cond = None
        self.cached_residual_uncond = None
        self.cached_J_z_norm = None
        self.cached_J_t_norm = None
        self.accumulated_skips = 0
        self.skip_this_step = False

    return [u.float() for u in x]


# =============================================================================
# Argument parsing & validation
# =============================================================================

def _validate_args(args):
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupported task: {args.task}"

    if args.sample_steps is None:
        args.sample_steps = 40 if "i2v" in args.task else 50

    if args.sample_shift is None:
        args.sample_shift = 5.0
        if "i2v" in args.task and args.size in ["832*480", "480*832"]:
            args.sample_shift = 3.0
        elif "flf2v" in args.task or "vace" in args.task:
            args.sample_shift = 16

    if args.frame_num is None:
        args.frame_num = 1 if "t2i" in args.task else 81

    if "t2i" in args.task:
        assert args.frame_num == 1, f"Unsupported frame_num {args.frame_num} for task {args.task}"

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)

    assert args.size in SUPPORTED_SIZES[args.task], \
        f"Unsupported size {args.size} for task {args.task}, supported: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an image or video from a text prompt or image using Wan")

    # I/O arguments
    parser.add_argument("--prompt_file", type=str, default=None,
                        help="JSONL file containing prompts. Overrides --prompt, --image, etc.")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save generated videos/images.")
    parser.add_argument("--save_file", type=str, default=None,
                        help="Output file path (only used if --prompt_file is not set).")

    # Task & model arguments
    parser.add_argument("--task", type=str, default="t2v-14B",
                        choices=list(WAN_CONFIGS.keys()))
    parser.add_argument("--size", type=str, default="1280*720",
                        choices=list(SIZE_CONFIGS.keys()))
    parser.add_argument("--frame_num", type=int, default=None,
                        help="Number of frames to generate (should be 4n+1).")
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--offload_model", type=str2bool, default=None)

    # Parallelism
    parser.add_argument("--ulysses_size", type=int, default=1)
    parser.add_argument("--ring_size", type=int, default=1)
    parser.add_argument("--t5_fsdp", action="store_true", default=False)
    parser.add_argument("--t5_cpu", action="store_true", default=False)
    parser.add_argument("--dit_fsdp", action="store_true", default=False)

    # Input sources
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--first_frame", type=str, default=None)
    parser.add_argument("--last_frame", type=str, default=None)
    parser.add_argument("--src_video", type=str, default=None)
    parser.add_argument("--src_mask", type=str, default=None)
    parser.add_argument("--src_ref_images", type=str, default=None)

    # Prompt extension
    parser.add_argument("--use_prompt_extend", action="store_true", default=False)
    parser.add_argument("--prompt_extend_method", type=str, default="local_qwen",
                        choices=["dashscope", "local_qwen"])
    parser.add_argument("--prompt_extend_model", type=str, default=None)
    parser.add_argument("--prompt_extend_target_lang", type=str, default="zh",
                        choices=["zh", "en"])

    # Sampling
    parser.add_argument("--sample_solver", type=str, default='unipc',
                        choices=['unipc', 'dpm++'])
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument("--sample_shift", type=float, default=None)
    parser.add_argument("--sample_guide_scale", type=float, default=5.0)
    parser.add_argument("--base_seed", type=int, default=42)

    # Profiling
    parser.add_argument("--measure_latency", type=str2bool, default=True)
    parser.add_argument("--measure_flops", type=str2bool, default=True)

    # SenCache arguments
    parser.add_argument("--sencache_thresh_start", type=float, default=0.005,
                        help="SenCache: error threshold for early steps.")
    parser.add_argument("--sencache_thresh_main", type=float, default=0.07,
                        help="SenCache: error threshold for later steps.")
    parser.add_argument("--sencache_K", type=int, default=10,
                        help="SenCache: maximum consecutive steps to skip.")
    parser.add_argument("--no_sencache", action="store_true", default=False,
                        help="Disable SenCache (run standard inference).")

    args = parser.parse_args()
    _validate_args(args)
    return args


# =============================================================================
# Profiling helpers
# =============================================================================

def _profile_generate(fn, measure_latency=True, measure_flops=True):
    if not (measure_latency or measure_flops):
        return fn(), None, None

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()

    if measure_flops:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, with_flops=True) as prof:
            result = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        total_flops = sum(getattr(evt, "flops", 0) or 0 for evt in prof.key_averages())
        gflops = total_flops / 1e9 if total_flops > 0 else None
        return result, end - start, gflops

    result = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()
    return result, end - start if measure_latency else None, None


def _log_generation_perf(item_index, latency_s, gflops, measure_flops):
    if latency_s is None and gflops is None:
        return
    parts = []
    if latency_s is not None:
        parts.append(f"latency={latency_s * 1000:.2f} ms")
    if measure_flops:
        parts.append(f"gflops={gflops:.2f}" if gflops is not None else "gflops=N/A")
    logging.info(f"[Item {item_index}] generate perf: {', '.join(parts)}")


def _init_logging(rank):
    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s] [RANK {rank}] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)])


# =============================================================================
# SenCache model patching
# =============================================================================

def _apply_sencache(pipeline, args, rank):
    """Monkey-patch the model with SenCache forward pass and Jacobian data."""
    if "t2v" not in args.task and "t2i" not in args.task:
        logging.warning(f"Task {args.task} does not support SenCache. Running standard inference.")
        return False

    logging.info(f"Rank {rank}: Applying SenCache patches...")

    try:
        # Load pre-computed Jacobian sensitivity data (conditional only)
        jacobian_path = "./sensitivity_wan21.npz"
        data = np.load(jacobian_path)
    except FileNotFoundError as e:
        logging.error(f"FATAL: Jacobian .npz file not found: {e.filename}")
        raise

    SCALING_FACTOR = 7093.614029533887
    threshold_start = args.sencache_thresh_start * SCALING_FACTOR
    threshold_main = args.sencache_thresh_main * SCALING_FACTOR

    model = pipeline.model
    model.__class__.forward = sencache_forward

    # Jacobian norms (conditional only — used for both cond and uncond decisions)
    model.J_z_norm = data['J_x_norm'].tolist()
    model.J_t_norm = data['J_t_norm'].tolist()
    model.sencache_timesteps_array = data['timesteps']

    # Thresholds
    model.sencache_threshold_start = threshold_start
    model.sencache_threshold_main = threshold_main
    model.sencache_K = args.sencache_K

    # Cache state
    model.cached_z = None              # last computed conditional z_t
    model.cached_t = None              # last computed conditional t
    model.cached_residual_cond = None  # residual from last conditional pass
    model.cached_residual_uncond = None  # residual from last unconditional pass
    model.cached_J_z_norm = None
    model.cached_J_t_norm = None
    model.accumulated_skips = 0
    model.skip_this_step = False       # decision flag shared between cond/uncond

    # Counters
    model.cnt = 0
    model.current_skip_count = 0
    model.all_skip_counts = []

    # Step boundaries
    total_calls = args.sample_steps * 2
    model.num_steps = total_calls
    model.threshold_switch_step = int(round(total_calls * 0.2))

    retention_steps = 2  # 2 calls = 1 denoising step
    model.ret_steps = retention_steps
    model.cutoff_steps = total_calls - retention_steps

    logging.info(f"  SenCache config:")
    logging.info(f"    thresh_start = {args.sencache_thresh_start} (raw) -> {threshold_start:.2f} (scaled)")
    logging.info(f"    thresh_main  = {args.sencache_thresh_main} (raw) -> {threshold_main:.2f} (scaled)")
    logging.info(f"    max consecutive skips (K) = {model.sencache_K}")
    logging.info(f"    skip zone = steps {model.ret_steps} to {model.cutoff_steps - 1}")

    return True


# =============================================================================
# Prompt loading
# =============================================================================

def _load_prompts(args, rank, world_size):
    """Load prompts from file or CLI args, and shard across ranks."""
    all_prompts = []

    if args.prompt_file:
        logging.info(f"Rank {rank} loading prompts from {args.prompt_file}")
        with open(args.prompt_file, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "prompt" not in data:
                        data = {"prompt": line}
                except json.JSONDecodeError:
                    data = {"prompt": line}
                data.setdefault("vbench_original_index", i)
                all_prompts.append(data)
        logging.info(f"Rank {rank} loaded {len(all_prompts)} prompts.")
    else:
        logging.info("No --prompt_file given, running in single-prompt mode.")
        single = {
            "vbench_original_index": "single",
            "prompt": args.prompt,
            "image": args.image,
            "first_frame": args.first_frame,
            "last_frame": args.last_frame,
            "src_video": args.src_video,
            "src_mask": args.src_mask,
            "src_ref_images": args.src_ref_images,
        }
        all_prompts = [single]

    # Shard across ranks
    my_prompts = all_prompts[rank::world_size]
    logging.info(f"Rank {rank} processing {len(my_prompts)} prompts (seed={args.base_seed}).")
    return my_prompts


# =============================================================================
# Pipeline creation
# =============================================================================

def _create_pipeline(args, cfg, device):
    """Create the appropriate Wan pipeline for the task."""
    common_kwargs = dict(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=0,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu,
    )

    if "t2v" in args.task or "t2i" in args.task:
        return wan.WanT2V(**common_kwargs)
    elif "i2v" in args.task:
        return wan.WanI2V(**common_kwargs)
    elif "flf2v" in args.task:
        return wan.WanFLF2V(**common_kwargs)
    elif "vace" in args.task:
        return wan.WanVace(**common_kwargs)
    else:
        raise ValueError(f"Unknown task type: {args.task}")


# =============================================================================
# Saving
# =============================================================================

def _save_output(video, args, item_prompt, item_index, cfg, rank, use_sencache):
    """Save a generated image or video to disk."""
    if video is None:
        return

    safe_prompt = "no_prompt"
    if item_prompt:
        safe_prompt = "".join(c for c in item_prompt if c.isalnum() or c in " _-").rstrip()[:50]

    suffix = '.png' if "t2i" in args.task else '.mp4'

    cache_tag = ""
    if use_sencache:
        cache_tag = (f"_SenCache_S{args.sencache_thresh_start}"
                     f"_M{args.sencache_thresh_main}"
                     f"_K{args.sencache_K}").replace(".", "")

    if args.output_dir:
        idx_str = f"{item_index:04d}" if isinstance(item_index, int) else str(item_index)
        save_file = os.path.join(args.output_dir, f"{idx_str}_{safe_prompt}{cache_tag}{suffix}")
    elif args.save_file and item_index == "single":
        save_file = args.save_file
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_file = f"{args.task}_{args.size.replace('*', 'x')}_{item_index}_{safe_prompt}_{ts}{suffix}"

    logging.info(f"Rank {rank}: Saving to {save_file}")

    if "t2i" in args.task:
        cache_image(tensor=video.squeeze(1)[None], save_file=save_file,
                    nrow=1, normalize=True, value_range=(-1, 1))
    else:
        cache_video(tensor=video[None], save_file=save_file,
                    fps=cfg.sample_fps, nrow=1, normalize=True, value_range=(-1, 1))

    logging.info(f"Rank {rank}: Saved {save_file}")


# =============================================================================
# Main generation
# =============================================================================

def generate(args):
    rank = int(os.getenv("SLURM_PROCID", os.getenv("RANK", 0)))
    world_size = int(os.getenv("SLURM_NTASKS", os.getenv("WORLD_SIZE", 1)))
    local_rank = int(os.getenv("SLURM_LOCALID", os.getenv("LOCAL_RANK", 0)))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = (world_size == 1)
        logging.info(f"offload_model not specified, set to {args.offload_model}.")

    if args.ulysses_size > 1 or args.ring_size > 1:
        logging.warning("Ulysses/Ring parallel is not supported in this independent-job mode.")
        assert args.ulysses_size * args.ring_size == world_size
        from xfuser.core.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    # Prompt extension
    prompt_expander = None
    if args.use_prompt_extend:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(
                model_name=args.prompt_extend_model,
                is_vl="i2v" in args.task or "flf2v" in args.task)
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(
                model_name=args.prompt_extend_model,
                is_vl="i2v" in args.task,
                device=rank)
        else:
            raise NotImplementedError(f"Unsupported prompt_extend_method: {args.prompt_extend_method}")

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    # Load prompts
    my_prompts = _load_prompts(args, rank, world_size)

    # Create output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # Create pipeline
    logging.info(f"Rank {rank}: Creating pipeline for task {args.task}")
    pipeline = _create_pipeline(args, cfg, device)

    # Apply SenCache (default unless --no_sencache)
    use_sencache = False
    if not args.no_sencache:
        use_sencache = _apply_sencache(pipeline, args, rank)
    else:
        logging.info(f"Rank {rank}: SenCache disabled. Running standard inference.")

    # --- Generation loop ---
    logging.info(f"Rank {rank} starting generation loop for {len(my_prompts)} items.")

    for prompt_idx, prompt_data in enumerate(my_prompts):
        item_prompt = prompt_data.get('prompt')
        item_index = prompt_data.get('vbench_original_index', 'unknown')
        item_seed = args.base_seed

        logging.info(f"Rank {rank} [{prompt_idx + 1}/{len(my_prompts)}]: "
                     f"index={item_index}, prompt='{str(item_prompt)[:50]}...'")

        try:
            video = None

            if "t2v" in args.task or "t2i" in args.task:
                if item_prompt is None:
                    logging.warning(f"Skipping item {item_index}: missing 'prompt'.")
                    continue

                current_prompt = item_prompt
                if prompt_expander:
                    logging.info(f"[Item {item_index}] Extending prompt...")
                    out = prompt_expander(current_prompt,
                                         tar_lang=args.prompt_extend_target_lang,
                                         seed=item_seed)
                    if out.status:
                        current_prompt = out.prompt
                    logging.info(f"[Item {item_index}] Extended: {current_prompt}")

                video, latency_s, gflops = _profile_generate(
                    lambda: pipeline.generate(
                        current_prompt,
                        size=SIZE_CONFIGS[args.size],
                        frame_num=args.frame_num,
                        shift=args.sample_shift,
                        sample_solver=args.sample_solver,
                        sampling_steps=args.sample_steps,
                        guide_scale=args.sample_guide_scale,
                        seed=item_seed,
                        offload_model=args.offload_model),
                    measure_latency=args.measure_latency,
                    measure_flops=args.measure_flops)
                _log_generation_perf(item_index, latency_s, gflops, args.measure_flops)

            elif "i2v" in args.task:
                item_image_path = prompt_data.get('image')
                if item_prompt is None or item_image_path is None:
                    logging.warning(f"Skipping item {item_index}: missing 'prompt' or 'image'.")
                    continue

                img = Image.open(item_image_path).convert("RGB")
                current_prompt = item_prompt
                if prompt_expander:
                    out = prompt_expander(current_prompt,
                                         tar_lang=args.prompt_extend_target_lang,
                                         image=img, seed=item_seed)
                    if out.status:
                        current_prompt = out.prompt

                video, latency_s, gflops = _profile_generate(
                    lambda: pipeline.generate(
                        current_prompt, img,
                        max_area=MAX_AREA_CONFIGS[args.size],
                        frame_num=args.frame_num,
                        shift=args.sample_shift,
                        sample_solver=args.sample_solver,
                        sampling_steps=args.sample_steps,
                        guide_scale=args.sample_guide_scale,
                        seed=item_seed,
                        offload_model=args.offload_model),
                    measure_latency=args.measure_latency,
                    measure_flops=args.measure_flops)
                _log_generation_perf(item_index, latency_s, gflops, args.measure_flops)

            elif "flf2v" in args.task:
                item_ff = prompt_data.get('first_frame')
                item_lf = prompt_data.get('last_frame')
                if item_prompt is None or item_ff is None or item_lf is None:
                    logging.warning(f"Skipping item {item_index}: missing 'prompt', 'first_frame', or 'last_frame'.")
                    continue

                first_frame = Image.open(item_ff).convert("RGB")
                last_frame = Image.open(item_lf).convert("RGB")
                current_prompt = item_prompt
                if prompt_expander:
                    out = prompt_expander(current_prompt,
                                         tar_lang=args.prompt_extend_target_lang,
                                         image=[first_frame, last_frame], seed=item_seed)
                    if out.status:
                        current_prompt = out.prompt

                video, latency_s, gflops = _profile_generate(
                    lambda: pipeline.generate(
                        current_prompt, first_frame, last_frame,
                        max_area=MAX_AREA_CONFIGS[args.size],
                        frame_num=args.frame_num,
                        shift=args.sample_shift,
                        sample_solver=args.sample_solver,
                        sampling_steps=args.sample_steps,
                        guide_scale=args.sample_guide_scale,
                        seed=item_seed,
                        offload_model=args.offload_model),
                    measure_latency=args.measure_latency,
                    measure_flops=args.measure_flops)
                _log_generation_perf(item_index, latency_s, gflops, args.measure_flops)

            elif "vace" in args.task:
                if item_prompt is None:
                    logging.warning(f"Skipping item {item_index}: missing 'prompt'.")
                    continue

                current_prompt = item_prompt
                if prompt_expander and args.use_prompt_extend != 'plain':
                    current_prompt = prompt_expander.forward(current_prompt)

                item_src_ref = prompt_data.get('src_ref_images')
                src_video, src_mask, src_ref_images = pipeline.prepare_source(
                    [prompt_data.get('src_video')],
                    [prompt_data.get('src_mask')],
                    [None if item_src_ref is None else item_src_ref.split(',')],
                    args.frame_num, SIZE_CONFIGS[args.size], device)

                video, latency_s, gflops = _profile_generate(
                    lambda: pipeline.generate(
                        current_prompt, src_video, src_mask, src_ref_images,
                        size=SIZE_CONFIGS[args.size],
                        frame_num=args.frame_num,
                        shift=args.sample_shift,
                        sample_solver=args.sample_solver,
                        sampling_steps=args.sample_steps,
                        guide_scale=args.sample_guide_scale,
                        seed=item_seed,
                        offload_model=args.offload_model),
                    measure_latency=args.measure_latency,
                    measure_flops=args.measure_flops)
                _log_generation_perf(item_index, latency_s, gflops, args.measure_flops)

            _save_output(video, args, item_prompt, item_index, cfg, rank, use_sencache)

        except Exception as e:
            logging.error(f"Rank {rank} FAILED on item {item_index}: {e}", exc_info=True)
            continue

    logging.info(f"Rank {rank} finished generation loop.")

    # --- Summary: skip statistics ---
    all_skip_counts = getattr(pipeline.model, 'all_skip_counts', None)
    if all_skip_counts and len(all_skip_counts) > 0:
        total_calls = args.sample_steps * 2
        avg_skips = np.mean(all_skip_counts)
        logging.info(f"[SenCache Summary] Rank {rank}: "
                     f"{len(all_skip_counts)} videos, "
                     f"avg skipped = {avg_skips:.1f}/{total_calls} calls "
                     f"({100 * avg_skips / total_calls:.1f}%)")
        logging.info(f"  Per-video skip counts: {all_skip_counts}")

        if args.output_dir:
            save_path = os.path.join(args.output_dir, f"rank_{rank}_sencache_stats.txt")
            with open(save_path, "w") as f:
                f.write(f"Rank: {rank}\n")
                f.write(f"Videos processed: {len(all_skip_counts)}\n")
                f.write(f"Average skipped calls per video: {avg_skips:.2f} / {total_calls}\n")
                f.write(f"Skip percentage: {100 * avg_skips / total_calls:.1f}%\n")
                f.write(f"Per-video skip counts: {all_skip_counts}\n")
            logging.info(f"Rank {rank}: Saved SenCache stats to {save_path}")
    elif use_sencache:
        logging.warning(f"Rank {rank}: No videos processed, no SenCache stats to report.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)