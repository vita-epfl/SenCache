# LTX-Video generation with SenCache (Jacobian-based sensitivity caching)
import argparse
import json
import logging
import os

import numpy as np
import torch
import torch.distributed as dist
from typing import Any, Dict, Optional, Tuple, Union

from diffusers import LTXPipeline
from diffusers.models.transformers import LTXVideoTransformer3DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_version,
    scale_lora_layers,
    unscale_lora_layers,
    export_to_video,
)

JACOBIAN_SCALING_FACTOR = 1015.9685034488028


# =============================================================================
# SenCache: Jacobian-based sensitivity caching for LTX-Video
# =============================================================================

def sencache_forward_ltx(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_attention_mask: torch.Tensor,
    num_frames: int,
    height: int,
    width: int,
    rope_interpolation_scale: Optional[Tuple[float, float, float]] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
) -> torch.Tensor:
    """
    Forward pass with SenCache for LTX-Video.

    In LTX, cond and uncond are batched together (batch dim 2), so the
    transformer is called once per denoising step. The caching decision is based
    on the CONDITIONAL sample (index 0) only.

    Uses a dual threshold strategy and a maximum consecutive skip limit (K).
    J norms are locked while skipping and refreshed when a full calculation happens.
    """
    # --- 1. Setup ---
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)

    image_rotary_emb = self.rope(hidden_states, num_frames, height, width, rope_interpolation_scale)

    if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
        encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

    # --- 2. SenCache: decide whether to skip this step ---
    z_t = hidden_states  # (B, S, C) tensor
    t_val = timestep
    should_calculate = True

    if self.cnt < self.threshold_switch_step:
        current_threshold = self.sencache_threshold_start
    else:
        current_threshold = self.sencache_threshold_main

    is_in_skip_zone = (self.cnt >= self.ret_steps) and (self.cnt < self.cutoff_steps)

    if is_in_skip_zone and self.cached_z is not None:
        # Decision based on conditional sample (index 0) only
        norm_delta_z = torch.norm(z_t[0] - self.cached_z).item()
        norm_delta_t = torch.abs(t_val[0] - self.cached_t).item()

        sensitivity_error = (self.cached_J_z_norm * norm_delta_z) + (self.cached_J_t_norm * norm_delta_t)

        if sensitivity_error < current_threshold and self.accumulated_skips < self.sencache_K:
            should_calculate = False
            self.accumulated_skips += 1
            self.current_skip_count += 1

    if should_calculate:
        self.accumulated_skips = 0

    # --- 3. Embeddings ---
    batch_size = hidden_states.shape[0]
    hidden_states = self.proj_in(hidden_states)
    ori_hidden_states_projected = hidden_states.clone()

    temb, embedded_timestep = self.time_embed(
        timestep.flatten(),
        batch_size=batch_size,
        hidden_dtype=hidden_states.dtype,
    )
    temb = temb.view(batch_size, -1, temb.size(-1))
    embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))
    encoder_hidden_states = self.caption_projection(encoder_hidden_states)
    encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.size(-1))

    # --- 4. Main computation: skip or calculate ---
    if not should_calculate:
        hidden_states = ori_hidden_states_projected + self.cached_residual
    else:
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                encoder_attention_mask=encoder_attention_mask,
            )

        scale_shift_values = self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift

        # Cache residual and update sensitivity state
        self.cached_residual = (hidden_states - ori_hidden_states_projected).detach()

        # Update reference point (conditional sample only)
        self.cached_z = z_t[0].detach()
        self.cached_t = t_val[0].detach()

        # Refresh J norms at current timestep
        current_t = t_val[0].item()
        nearest_index = np.argmin(np.abs(self.sencache_timesteps_array - current_t))
        self.cached_J_z_norm = self.J_z_norm[nearest_index]
        self.cached_J_t_norm = self.J_t_norm[nearest_index]

    # --- 5. Post-processing and counter ---
    output = self.proj_out(hidden_states)

    self.cnt += 1
    if self.cnt >= self.num_steps:
        self.all_skip_counts.append(self.current_skip_count)
        total_steps = self.num_steps
        logging.info(
            f"[SenCache] Generation complete: "
            f"skipped {self.current_skip_count}/{total_steps} steps "
            f"({100 * self.current_skip_count / total_steps:.1f}%)"
        )
        self.current_skip_count = 0
        self.cnt = 0
        self.cached_z = None
        self.cached_t = None
        self.cached_residual = None
        self.cached_J_z_norm = None
        self.cached_J_t_norm = None
        self.accumulated_skips = 0

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


# =============================================================================
# SenCache model patching
# =============================================================================

def _apply_sencache(pipe, args, rank):
    """Monkey-patch the LTX transformer with SenCache."""
    logging.info(f"Rank {rank}: Applying SenCache patches...")

    try:
        jacobian_path = "./sensitivity_ltx.npz"
        data = np.load(jacobian_path)
    except FileNotFoundError as e:
        logging.error(f"FATAL: Jacobian .npz file not found: {e.filename}")
        raise

    model = pipe.transformer
    model.__class__.forward = sencache_forward_ltx

    # Jacobian norms (conditional only)
    model.J_z_norm = data['J_x_norm'].tolist()
    model.J_t_norm = data['J_t_norm'].tolist()
    model.sencache_timesteps_array = data['timesteps']

    # Thresholds
    threshold_start = args.sencache_thresh_start * JACOBIAN_SCALING_FACTOR
    threshold_main = args.sencache_thresh_main * JACOBIAN_SCALING_FACTOR
    model.sencache_threshold_start = threshold_start
    model.sencache_threshold_main = threshold_main
    model.sencache_K = args.sencache_K

    # Step boundaries
    total_steps = args.num_inference_steps
    model.num_steps = total_steps
    model.threshold_switch_step = int(round(total_steps * 0.2))
    retention_steps = int(round(total_steps * 0.02))
    model.ret_steps = retention_steps
    model.cutoff_steps = total_steps - retention_steps

    # Cache state
    model.cached_z = None
    model.cached_t = None
    model.cached_residual = None
    model.cached_J_z_norm = None
    model.cached_J_t_norm = None
    model.accumulated_skips = 0

    # Counters
    model.cnt = 0
    model.current_skip_count = 0
    model.all_skip_counts = []

    logging.info(f"  SenCache config:")
    logging.info(f"    thresh_start = {args.sencache_thresh_start} (raw) -> {threshold_start:.2f} (scaled)")
    logging.info(f"    thresh_main  = {args.sencache_thresh_main} (raw) -> {threshold_main:.2f} (scaled)")
    logging.info(f"    max consecutive skips (K) = {model.sencache_K}")
    logging.info(f"    skip zone = steps {model.ret_steps} to {model.cutoff_steps - 1}")


# =============================================================================
# Prompt loading
# =============================================================================

def _load_prompts(args, rank, world_size):
    """Load prompts from file or CLI args, broadcast and shard across ranks."""
    all_prompts = []

    if args.prompt_file:
        if rank == 0:
            logging.info(f"Loading prompts from {args.prompt_file}")
            try:
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
            except Exception as e:
                logging.error(f"Failed to read prompt file: {e}")
                all_prompts = None

        broadcast_list = [all_prompts]
        if world_size > 1:
            dist.broadcast_object_list(broadcast_list, src=0)
        all_prompts = broadcast_list[0]

        if all_prompts is None or not all_prompts:
            return None
    else:
        all_prompts = [{
            "prompt": args.prompt,
            "vbench_original_index": "single",
        }]

    my_prompts = all_prompts[rank::world_size]
    logging.info(f"Rank {rank} processing {len(my_prompts)} prompts (seed={args.seed}).")
    return my_prompts


# =============================================================================
# Saving
# =============================================================================

def _save_output(video, args, item_prompt, item_index, rank, use_sencache):
    """Save a generated video to disk."""
    if video is None:
        return

    safe_prompt = "".join(c for c in item_prompt if c.isalnum() or c in " _-").rstrip()[:50]

    cache_tag = "_Standard"
    if use_sencache:
        cache_tag = (f"_SenCache_S{args.sencache_thresh_start}"
                     f"_M{args.sencache_thresh_main}"
                     f"_K{args.sencache_K}").replace(".", "")

    idx_str = f"{item_index:04d}" if isinstance(item_index, int) else str(item_index)
    video_name = f"{idx_str}_{safe_prompt}{cache_tag}.mp4"
    video_path = os.path.join(args.output_dir, video_name)

    export_to_video(video, video_path, fps=args.fps)
    logging.info(f"Rank {rank}: Saved {video_path}")


# =============================================================================
# Main
# =============================================================================

def main(args):
    # --- Distributed setup ---
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"

    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s] RANK {rank} %(levelname)s: %(message)s",
    )

    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(device)
    else:
        logging.info("Running in non-distributed mode.")

    # --- Load prompts ---
    my_prompts = _load_prompts(args, rank, world_size)
    if my_prompts is None:
        logging.error("Exiting due to prompt loading error.")
        if world_size > 1:
            dist.destroy_process_group()
        return

    # --- Load model ---
    logging.info(f"Rank {rank} loading model from {args.ckpts_path}...")
    pipe = LTXPipeline.from_pretrained(args.ckpts_path, torch_dtype=torch.bfloat16)
    pipe.to(device)

    # --- Apply SenCache (default unless --no_sencache) ---
    use_sencache = False
    if not args.no_sencache:
        try:
            _apply_sencache(pipe, args, rank)
            use_sencache = True
        except FileNotFoundError:
            if world_size > 1:
                dist.destroy_process_group()
            return
    else:
        logging.info(f"Rank {rank}: SenCache disabled. Running standard inference.")

    # --- Generation loop ---
    logging.info(f"Rank {rank} starting generation loop for {len(my_prompts)} items.")

    for item_idx, item_data in enumerate(my_prompts):
        item_prompt = item_data["prompt"]
        item_index = item_data["vbench_original_index"]

        logging.info(f"Rank {rank} [{item_idx + 1}/{len(my_prompts)}]: "
                     f"index={item_index}, prompt='{str(item_prompt)[:50]}...'")

        generator = torch.Generator(device).manual_seed(args.seed)

        try:
            video = pipe(
                prompt=item_prompt,
                negative_prompt=args.negative_prompt,
                width=args.width,
                height=args.height,
                num_frames=args.num_frames,
                decode_timestep=args.decode_timestep,
                decode_noise_scale=args.decode_noise_scale,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
            ).frames[0]

            _save_output(video, args, item_prompt, item_index, rank, use_sencache)

        except Exception as e:
            logging.error(f"Rank {rank} FAILED on item {item_index}: {e}", exc_info=True)
            continue

    logging.info(f"Rank {rank} finished generation loop.")

    # --- Summary: skip statistics ---
    all_skip_counts = getattr(pipe.transformer, 'all_skip_counts', None)
    if all_skip_counts and len(all_skip_counts) > 0:
        total_steps = args.num_inference_steps
        avg_skips = np.mean(all_skip_counts)
        logging.info(f"[SenCache Summary] Rank {rank}: "
                     f"{len(all_skip_counts)} videos, "
                     f"avg skipped = {avg_skips:.1f}/{total_steps} steps "
                     f"({100 * avg_skips / total_steps:.1f}%)")
        logging.info(f"  Per-video skip counts: {all_skip_counts}")
    elif use_sencache:
        logging.warning(f"Rank {rank}: No videos processed, no SenCache stats to report.")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LTX-Video generation with SenCache")

    # I/O
    parser.add_argument("--prompt_file", type=str, default=None,
                        help="JSONL or text file containing prompts.")
    parser.add_argument("--output_dir", type=str, default="sencache_ltx_results")
    parser.add_argument("--prompt", type=str, default="A beautiful sunset over the ocean",
                        help="Prompt (if --prompt_file is not set).")
    parser.add_argument("--negative_prompt", type=str,
                        default="worst quality, inconsistent motion, blurry, jittery, distorted")

    # Model
    parser.add_argument("--ckpts_path", type=str, default="a-r-r-o-w/LTX-Video-0.9.1-diffusers")

    # Generation
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--num_frames", type=int, default=161)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--decode_timestep", type=float, default=0.03)
    parser.add_argument("--decode_noise_scale", type=float, default=0.025)

    # SenCache
    parser.add_argument("--sencache_thresh_start", type=float, default=0.01,
                        help="SenCache: error threshold for early steps.")
    parser.add_argument("--sencache_thresh_main", type=float, default=0.7,
                        help="SenCache: error threshold for later steps.")
    parser.add_argument("--sencache_K", type=int, default=4,
                        help="SenCache: maximum consecutive steps to skip.")
    parser.add_argument("--no_sencache", action="store_true", default=False,
                        help="Disable SenCache (run standard inference).")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)