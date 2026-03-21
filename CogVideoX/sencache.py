# CogVideoX generation with SenCache (Jacobian-based sensitivity caching)
import argparse
import json
import logging
import math
import os

import numpy as np
import torch
import torch.distributed as dist
from typing import Any, Dict, Optional, Tuple, Union

from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_version,
    scale_lora_layers,
    unscale_lora_layers,
    export_to_video,
    load_image,
)
from diffusers import CogVideoXPipeline, CogVideoXImageToVideoPipeline

JACOBIAN_SCALING_FACTOR = 1099.8181667894016 #set based on the shape of the latent to denormalize


# =============================================================================
# SenCache: Jacobian-based sensitivity caching for CogVideoX
# =============================================================================

def sencache_forward_cogvid(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: Union[int, float, torch.LongTensor],
    timestep_cond: Optional[torch.Tensor] = None,
    ofs: Optional[Union[int, float, torch.LongTensor]] = None,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
):
    """
    Forward pass with SenCache for CogVideoX.

    In CogVideoX, cond and uncond are batched together (batch dim 2), so the
    transformer is called once per denoising step. The caching decision is based
    on the CONDITIONAL sample (index 0) only.

    Uses a dual threshold strategy and a maximum consecutive skip limit (K).
    J norms are locked while skipping and refreshed when a full calculation happens.
    """
    # --- 1. Setup and Embeddings ---
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0
    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)

    t_emb = self.time_proj(timestep).to(dtype=hidden_states.dtype)
    emb = self.time_embedding(t_emb, timestep_cond)
    if self.ofs_embedding is not None:
        ofs_emb = self.ofs_proj(ofs).to(dtype=hidden_states.dtype)
        emb = emb + self.ofs_embedding(ofs_emb)

    # --- 2. Patch embedding ---
    z_t = hidden_states   # VAE latent input, shape [2, C, F, H, W]
    t_val = timestep      # Time value, shape [2] (e.g., [981., 981.])

    hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
    hidden_states = self.embedding_dropout(hidden_states)
    text_seq_length = encoder_hidden_states.shape[1]
    encoder_hidden_states = hidden_states[:, :text_seq_length]
    hidden_states = hidden_states[:, text_seq_length:]

    # --- 3. SenCache: decide whether to skip this step ---
    # Decision based on CONDITIONAL sample only (index 0 in batch)
    should_calculate = True

    if self.cnt < self.threshold_switch_step:
        current_threshold = self.sencache_threshold_start
    else:
        current_threshold = self.sencache_threshold_main

    is_in_skip_zone = (self.cnt >= self.ret_steps) and (self.cnt < self.cutoff_steps)

    if is_in_skip_zone and self.cached_z is not None:
        # Use conditional sample (index 0) for the caching decision
        norm_delta_z = torch.norm(z_t[0] - self.cached_z).item()
        norm_delta_t = torch.abs(t_val[0] - self.cached_t).item()

        sensitivity_error = (self.cached_J_z_norm * norm_delta_z) + (self.cached_J_t_norm * norm_delta_t)

        if sensitivity_error < current_threshold and self.accumulated_skips < self.sencache_K:
            should_calculate = False
            self.accumulated_skips += 1
            self.current_skip_count += 1

    if should_calculate:
        self.accumulated_skips = 0

    # --- 4. Main computation: skip or calculate ---
    ori_hidden_states = hidden_states.clone()
    ori_encoder_hidden_states = encoder_hidden_states.clone()

    if not should_calculate and self.cached_residual is not None:
        hidden_states += self.cached_residual
        encoder_hidden_states += self.cached_residual_encoder
    else:
        for i, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states, encoder_hidden_states, emb, image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                )

        # Cache residuals and update sensitivity state
        self.cached_residual = (hidden_states - ori_hidden_states).detach()
        self.cached_residual_encoder = (encoder_hidden_states - ori_encoder_hidden_states).detach()

        # Update reference point (conditional sample only)
        self.cached_z = z_t[0].detach()
        self.cached_t = t_val[0].detach()

        # Refresh J norms at current timestep
        current_t = t_val[0].item()
        nearest_index = np.argmin(np.abs(self.sencache_timesteps_array - current_t))
        self.cached_J_z_norm = self.J_z_norm[nearest_index]
        self.cached_J_t_norm = self.J_t_norm[nearest_index]

    # --- 5. Post-processing and unpatchify ---
    batch_size, num_frames, channels, height, width = z_t.shape
    if not self.config.use_rotary_positional_embeddings:
        hidden_states = self.norm_final(hidden_states)
    else:
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        hidden_states = self.norm_final(hidden_states)
        hidden_states = hidden_states[:, text_seq_length:]

    hidden_states = self.norm_out(hidden_states, temb=emb)
    hidden_states = self.proj_out(hidden_states)

    p = self.config.patch_size
    p_t = self.config.patch_size_t
    if p_t is None:
        output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
    else:
        output = hidden_states.reshape(
            batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
        )
        output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    # --- 6. Step counter and reset ---
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
        self.cached_residual_encoder = None
        self.accumulated_skips = 0
        self.cached_J_z_norm = None
        self.cached_J_t_norm = None

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


# =============================================================================
# SenCache model patching
# =============================================================================

def _apply_sencache(pipe, args, rank):
    """Monkey-patch the CogVideoX transformer with SenCache."""
    logging.info(f"Rank {rank}: Applying SenCache patches...")

    try:
        jacobian_path = "./sensitivity_cogvid.npz"
        data = np.load(jacobian_path)
    except FileNotFoundError as e:
        logging.error(f"FATAL: Jacobian .npz file not found: {e.filename}")
        raise

    model = pipe.transformer
    model.__class__.forward = sencache_forward_cogvid

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
    model.ret_steps = 1
    model.cutoff_steps = total_steps - 1

    # Cache state
    model.cached_z = None
    model.cached_t = None
    model.cached_residual = None
    model.cached_residual_encoder = None
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
            "image_path": args.image_path,
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
    if args.generate_type == "t2v":
        pipe = CogVideoXPipeline.from_pretrained(args.ckpts_path, torch_dtype=torch.bfloat16)
    else:
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(args.ckpts_path, torch_dtype=torch.bfloat16)
    pipe.to(device)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

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
            if args.generate_type == "t2v":
                video = pipe(
                    prompt=item_prompt,
                    negative_prompt=args.negative_prompt,
                    width=args.width,
                    height=args.height,
                    num_frames=args.num_frames,
                    use_dynamic_cfg=True,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    generator=generator,
                ).frames[0]
            else:
                image_path = item_data.get("image_path") or args.image_path
                if image_path is None:
                    logging.warning(f"Skipping item {item_index} (i2v): missing image path.")
                    continue
                image = load_image(image=image_path)
                video = pipe(
                    prompt=item_prompt,
                    negative_prompt=args.negative_prompt,
                    image=image,
                    width=args.width,
                    height=args.height,
                    num_frames=args.num_frames,
                    use_dynamic_cfg=True,
                    guidance_scale=args.guidance_scale,
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
    parser = argparse.ArgumentParser(description="CogVideoX generation with SenCache")

    # I/O
    parser.add_argument("--prompt_file", type=str, default=None,
                        help="JSONL or text file containing prompts.")
    parser.add_argument("--output_dir", type=str, default="sencache_results")
    parser.add_argument("--prompt", type=str, default="A clear, turquoise river...",
                        help="Prompt (if --prompt_file is not set).")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--image_path", type=str, default=None,
                        help="Input image path (for i2v).")

    # Model
    parser.add_argument("--ckpts_path", type=str, default="THUDM/CogVideoX1.5-5B")
    parser.add_argument("--generate_type", type=str, default="t2v", choices=["t2v", "i2v"])

    # Generation
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--fps", type=int, default=8)

    # SenCache
    parser.add_argument("--sencache_thresh_start", type=float, default=0.005,
                        help="SenCache: error threshold for early steps.")
    parser.add_argument("--sencache_thresh_main", type=float, default=0.07,
                        help="SenCache: error threshold for later steps.")
    parser.add_argument("--sencache_K", type=int, default=10,
                        help="SenCache: maximum consecutive steps to skip.")
    parser.add_argument("--no_sencache", action="store_true", default=False,
                        help="Disable SenCache (run standard inference).")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)