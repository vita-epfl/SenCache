import argparse
import logging
import os
import sys
import warnings
import random
from pathlib import Path
import json
import math
import datetime as dt  # DDP Timeout

import torch
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from PIL import Image

# --- Diffusers (LTX) Imports ---
from diffusers import LTXPipeline
import transformers

# --- Data Loading Imports ---
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

try:
    from decord import VideoReader, cpu
except ImportError:
    print("Error: 'decord' library not found. Please install it with 'pip install decord'")
    sys.exit(1)

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------
# Aggregation + Save
# ---------------------------------------------------------------
def aggregate_and_save(local_sum_sensitivities, local_sample_counts, local_t_indices_original,
                       timesteps_all, device, args, is_main_process, checkpoint=False):
    """Aggregates data from all processes and saves the current means."""
    

    # Tensors are initialized to the FULL size (args.num_t_steps)
    all_sums_tensor = torch.zeros((args.num_t_steps, 2), device=device, dtype=torch.float32)
    all_counts_tensor = torch.zeros(args.num_t_steps, device=device, dtype=torch.float32)

    # Place local results (which are also full-sized) into the tensors
    local_t_indices_tensor = torch.tensor(local_t_indices_original, device=device, dtype=torch.long)
    all_sums_tensor[local_t_indices_tensor] = torch.tensor(local_sum_sensitivities, device=device, dtype=torch.float32)
    all_counts_tensor[local_t_indices_tensor] = torch.tensor(local_sample_counts, device=device, dtype=torch.float32)

    dist.all_reduce(all_sums_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(all_counts_tensor, op=dist.ReduceOp.SUM)

    if is_main_process:
        current_mean_sensitivities = all_sums_tensor / (all_counts_tensor.unsqueeze(-1) + 1e-8)
        current_means_np = current_mean_sensitivities.cpu().numpy()

        model_name_safe = args.ckpts_path.replace('/', '_')
        filename = (f"sensitivity_checkpoint_{model_name_safe}_t{args.num_t_steps}_s{args.max_samples}.npz"
                    if checkpoint else
                    f"sensitivity_results_{model_name_safe}_t{args.num_t_steps}_s{args.max_samples}.npz")
        output_path = Path(args.output_dir) / filename
        np.savez(
            output_path,
            timesteps=timesteps_all,
            J_x_norm=current_means_np[:, 0],
            J_t_norm=current_means_np[:, 1],
            sample_counts=all_counts_tensor.cpu().numpy()
        )
        print(f"\nSaved {'Checkpoint' if checkpoint else 'Final results'} to: {output_path}")

    dist.barrier()

# ---------------------------------------------------------------
# Arg Parsing
# ---------------------------------------------------------------
def _validate_args(args):
    assert args.ckpts_path is not None, "Please specify the checkpoint path (--ckpts_path)."
    assert args.json_path is not None, "Please specify --json_path"
    assert args.video_base_path is not None, "Please specify --video_base_path"
    assert os.path.exists(args.json_path), f"JSON file not found at: {args.json_path}"
    assert os.path.exists(args.video_base_path), f"Video base path not found at: {args.video_base_path}"
    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate sensitivity (finite Jacobian estimates) for an LTX model from a JSON dataset."
    )
    # --- LTX Args ---
    parser.add_argument('--ckpts_path', type=str, default="a-r-r-o-w/LTX-Video-0.9.1-diffusers",
                        help='HF repo or local path to LTX checkpoint')
    parser.add_argument('--height', type=int, default=512, help='Pixel height (divisible by 32).')
    parser.add_argument('--width', type=int, default=768, help='Pixel width (divisible by 32).')
    parser.add_argument('--frame_num', type=int, default=161, help='Frames (recommended 8*k+1, e.g., 161).')
    parser.add_argument('--mu', type=float, default=3.2, help='Mu value for dynamic shifting in the scheduler.')

    # --- Dataset Args ---
    parser.add_argument("--json_path", type=str, required=True, help="Path to the .json with video paths and text.")
    parser.add_argument("--video_base_path", type=str, required=True, help="Base directory for videos.")
    parser.add_argument("--num_workers", type=int, default=6, help="DataLoader workers.")

    # --- Sensitivity Args ---
    parser.add_argument("--num_t_steps", type=int, default=50, help="Timesteps to evaluate.")
    parser.add_argument("--max_samples", type=int, default=2048, help="Total samples to process.")
    parser.add_argument("--output_dir", type=str, default="./results_sensitivity_ltx", help="Save directory.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU.")
    parser.add_argument("--base_seed", type=int, default=42, help="Base RNG seed.")

    args = parser.parse_args()
    _validate_args(args)
    return args

# ---------------------------------------------------------------
# Logging
# ---------------------------------------------------------------
def _init_logging(rank):
    logging.basicConfig(
        level=logging.INFO,
        format=f"[rank{rank}] %(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )
    logging.getLogger('PIL').setLevel(logging.ERROR)
    logging.getLogger('transformers').setLevel(logging.ERROR)

# ---------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------
class VideoTextJSONDataset(Dataset):
    """
    Loads videos and text from the provided JSON file.
    Returns: (video_tensor, text_string)
    """
    def __init__(self, json_path, video_base_path, size, frame_num):
        super().__init__()
        try:
            self.data = json.load(open(json_path))
        except Exception as e:
            raise ValueError(f"Error loading JSON file {json_path}: {e}")
            
        self.base_path = Path(video_base_path)
        self.frame_num = frame_num

        if isinstance(size, tuple):
            pass
        elif isinstance(size, str):
            H, W = map(int, size.split('*'))
            size = (H, W)
        else:
            raise ValueError(f"Invalid size format: {size}")

        self.transform = transforms.Resize(size, antialias=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            idx = idx % len(self.data)
            item = self.data[idx]

            text = item['text']
            video_path = self.base_path / item['file_path']

            vr = VideoReader(str(video_path), ctx=cpu(0))
            total_frames = len(vr)
            indices = np.linspace(0, total_frames - 1, self.frame_num, dtype=int)
            frames = vr.get_batch(indices).asnumpy()  # (F, H, W, C)

            frames_th = torch.tensor(frames).permute(0, 3, 1, 2)  # (F, C, H, W)
            frames_resized = self.transform(frames_th)            # (F, C, Hn, Wn)
            video_tensor = (frames_resized.float() / 255.0) * 2.0 - 1.0  # [-1, 1]
            video_tensor = video_tensor.permute(1, 0, 2, 3)       # (C, F, H, W)

            return video_tensor, text
        except Exception as e:
            logging.warning(f"Error loading item {idx} ({video_path}): {e}. Skipping.")
            # Return the next item on error
            return self.__getitem__((idx + 1) % len(self))


def calculate_solver_step_sensitivity(
    model, sample_scheduler, x_t, t, t_next, y, transformer_kwargs, device, generator_seed
):
    """
    ||f(x_{t-1}, t) - f(x_t, t)|| / ||x_{t-1} - x_t||
    """
    g = torch.Generator(device=device).manual_seed(generator_seed)

    def model_fn(x_in_tensor, t_in_tensor):
        # --- Model function for 3D Transformer (SiT/Latte) ---
        # x_in_tensor is (N, C)
        x_in_batch = x_in_tensor.to(torch.float32).unsqueeze(0)  # (1, N, C)
        t_in_int = t_in_tensor.to(torch.int) # LTX needs integer timesteps
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # Model expects (B, N, C) and kwargs
            pred = model(hidden_states=x_in_batch, timestep=t_in_int, **{**y, **transformer_kwargs}).sample
        return pred.squeeze(0).to(torch.float32) # Return (N, C)

    # Get model prediction at time t
    pred_t = model_fn(x_t, t)

    # Use scheduler to get x at time t-1 (t_next)
    x_t_next_batch = sample_scheduler.step(
        pred_t.unsqueeze(0),
        t.to(torch.int),
        x_t.unsqueeze(0),
        return_dict=False,
        generator=g
    )[0]
    x_t_next = x_t_next_batch.squeeze(0)

    # Get model prediction at time t-1
    pred_t_next = model_fn(x_t_next, t) 

    with torch.no_grad():
        delta_x_norm = torch.norm(x_t_next - x_t).item()
        delta_pred_norm = torch.norm(pred_t_next - pred_t).item()

    if delta_x_norm < 1e-8:
        return 0.0
    return delta_pred_norm / delta_x_norm


def calculate_jacobian_norm_T(model, x, t, y, t_next, transformer_kwargs, device):
    """
    || (f(x_t, t_next) - f(x_t, t)) / (t_next - t) ||
    """
    x_batch = x.unsqueeze(0).to(device, dtype=torch.float32)
    t = t.to(device, dtype=torch.float32)
    t_next = t_next.to(device, dtype=torch.float32)

    def model_fn_t(t_in):
        # --- Model function for LTX Transformer ---
        t_in_int = t_in.to(torch.int)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = model(hidden_states=x_batch, timestep=t_in_int, **{**y, **transformer_kwargs}).sample
        return out.to(torch.float32)

    with torch.no_grad():
        f_t = model_fn_t(t)
        f_tn = model_fn_t(t_next)
        
        # t_next is the *smaller* timestep, so (t_next - t) is negative
        dt = t_next.item() - t.item()
        
        if abs(dt) < 1e-8:
            # Avoid division by zero if t and t_next are the same
            return 0.0
            
        J_t_vec = (f_tn - f_t) / dt
        del f_tn, f_t

    return J_t_vec.norm().item()

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def run_sensitivity(args):
    # --- DDP Setup ---
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    _init_logging(rank)
    is_main_process = (rank == 0)

    ddp_timeout = dt.timedelta(minutes=120)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        timeout=ddp_timeout
    )

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]
    torch.manual_seed(args.base_seed + rank)
    np.random.seed(args.base_seed + rank)
    random.seed(args.base_seed + rank)

    # --- Load LTX pipeline ---
    if is_main_process: logging.info(f"Creating LTX pipeline from: {args.ckpts_path}")
    Image.MAX_IMAGE_PIXELS = None
    transformers.utils.logging.set_verbosity_error()

    # Load in bfloat16 for speed/memory
    pipe = LTXPipeline.from_pretrained(args.ckpts_path, torch_dtype=torch.bfloat16)
    pipe.to(device)
    if is_main_process: logging.info("Pipeline loaded.")

    # Components
    model_to_test = pipe.transformer
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    sample_scheduler = pipe.scheduler

    # --- Robust compression ratio fallback ---
    spatial_ratio = getattr(vae.config, "spatial_compression_ratio", None)
    temporal_ratio = getattr(vae.config, "temporal_compression_ratio", None)
    spatial_ratio = int(spatial_ratio) if isinstance(spatial_ratio, (int, float)) and spatial_ratio else 32
    temporal_ratio = int(temporal_ratio) if isinstance(temporal_ratio, (int, float)) and temporal_ratio else 8
    
    if is_main_process:
        logging.info(f"Vae ratios: spatial={spatial_ratio}, temporal={temporal_ratio}")

    # --- Prepare Data ---
    size_tuple = (args.height, args.width)
    if is_main_process: logging.info(f"Target pixel size: {size_tuple}, Frames: {args.frame_num}")
    if (args.height % spatial_ratio) or (args.width % spatial_ratio) or ((args.frame_num - 1) % temporal_ratio):
        logging.warning(
            f"Recommended: H/W divisible by {spatial_ratio} and (frames-1) divisible by {temporal_ratio}. "
            f"Got H={args.height}, W={args.width}, F={args.frame_num}."
        )

    dataset = VideoTextJSONDataset(
        json_path=args.json_path,
        video_base_path=args.video_base_path,
        size=size_tuple,
        frame_num=args.frame_num
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.base_seed)

    if args.max_samples % (world_size * args.batch_size) != 0:
        logging.warning(f"Max samples ({args.max_samples}) not divisible by total batch size ({world_size * args.batch_size}).")
    max_samples_per_gpu = math.ceil(args.max_samples / world_size)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    if is_main_process:
        logging.info(f"Dataset size: {len(dataset)}. Sampler size per GPU: {len(sampler)}")
        logging.info(f"Total max samples: {args.max_samples}. Max samples per GPU: {max_samples_per_gpu}")

    # --- Prepare Timesteps (Scheduler) ---
    sample_scheduler.set_timesteps(args.num_t_steps, device=device, mu=args.mu)
    timesteps_tensor = sample_scheduler.timesteps
    
    # Ensure they are sorted descending
    timesteps_tensor, _ = torch.sort(timesteps_tensor, descending=True)

    sigmas_tensor = sample_scheduler.sigmas
    sigmas_tensor, _ = torch.sort(sigmas_tensor, descending=True)
    
    # LTX scheduler might add an extra t=0 step. We remove it if it's
    # more than num_t_steps, as we can't shift t=0.
    if len(timesteps_tensor) > args.num_t_steps:
        if is_main_process:
            logging.info(f"Scheduler provided {len(timesteps_tensor)} steps. Truncating final step.")
        timesteps_tensor = timesteps_tensor[:args.num_t_steps]
        sigmas_tensor = sigmas_tensor[:args.num_t_steps]

    if is_main_process:
        logging.info(f"Actual number of timesteps being analyzed: {len(timesteps_tensor)}")

    timesteps_all = timesteps_tensor.cpu().numpy()
    sigmas_all = sigmas_tensor.cpu().numpy()

    if is_main_process:
        logging.info(f"Using {sample_scheduler.__class__.__name__} scheduler.")
        logging.info(f"Calculating for {len(timesteps_all)} steps. Range: [{timesteps_all.min():.2f}, {timesteps_all.max():.2f}]")

    # Shifted times for t_next (t_{i-1})
    shift_time = np.roll(timesteps_all, -1)
    shift_time[-1] = 0 # Last step shifts to 0

    # All GPUs process all timesteps
    local_timesteps = timesteps_all
    local_shift_time = shift_time
    local_sigmas = sigmas_all
    local_t_indices_original = list(range(len(timesteps_all)))


    # --- Loop state ---
    local_sum_sensitivities = np.zeros((len(timesteps_all), 2), dtype=np.float32)
    local_sample_counts = np.zeros(len(timesteps_all), dtype=np.float32)
    total_samples_processed_gpu = 0
    data_iter = iter(loader)
    sampler.set_epoch(0)

    if is_main_process:
        logging.info(f"Starting sensitivity calculation on {world_size} GPUs.")
        logging.info(f"Each GPU will process ALL {len(local_timesteps)} timesteps for each sample.")

    # --- Main Loop ---
    while total_samples_processed_gpu < max_samples_per_gpu:
        if is_main_process:
            logging.info(f"\nTop of main loop. Processed {total_samples_processed_gpu}/{max_samples_per_gpu}")
        try:
            video_batch, text_batch = next(data_iter)
        except StopIteration:
            if is_main_process: logging.info("Restarting data iterator...")
            sampler.set_epoch(getattr(sampler, "epoch", 0) + 1)
            data_iter = iter(loader)
            try:
                video_batch, text_batch = next(data_iter)
            except StopIteration:
                logging.warning("DataLoader empty even after reset. Ending.")
                break

        # 1. Encode video to get "clean" latents
        #    We use the VAE to encode the real video data.
        video_batch = video_batch.to(device, dtype=pipe.dtype)
        with torch.no_grad():
            # vae.encode returns a distribution. We sample from it.
            # LTX vae expects (B, C, F, H, W)
            latents_batch_dist = vae.encode(video_batch).latent_dist
            latents_batch = latents_batch_dist.sample()
            # Apply scaling factor, standard for diffusers VAEs
            latents_batch = latents_batch * vae.config.scaling_factor
            
            # 2. Encode text
            toks = tokenizer(
                list(text_batch),
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            # both CLIPTextModel and T5EncoderModel return last hidden state at index 0
            prompt_embeds_batch = text_encoder(toks.input_ids.to(device))[0].to(pipe.dtype)
            attention_mask_batch = toks.attention_mask.to(device)
            
        b = len(text_batch)

        batch_loop_desc = f"Rank {rank} (Processed: {total_samples_processed_gpu}/{max_samples_per_gpu})"
        batch_loop = tqdm(range(b), desc=batch_loop_desc, disable=(not is_main_process))

        samples_processed_in_batch = 0
        for i in batch_loop:
            if (total_samples_processed_gpu + samples_processed_in_batch) >= max_samples_per_gpu:
                break


            x_i_raw = latents_batch[i].to(torch.float32)



            _c, _f, _h, _w = x_i_raw.shape
            
            # Define transformer_kwargs *dynamically* for this specific sample.
            transformer_kwargs_i = {
                'num_frames': _f,
                'height': _h,
                'width': _w
            }

            x_i_permuted = x_i_raw.permute(1, 2, 3, 0).contiguous() # (F, H, W, C)
            x_i_clean = x_i_permuted.reshape(-1, _c) # (N, C) where N = F*H*W

            # text conditioning
            context_i = prompt_embeds_batch[i].unsqueeze(0)
            mask_i = attention_mask_batch[i].unsqueeze(0)
            y_i = {'encoder_hidden_states': context_i, 'encoder_attention_mask': mask_i}


            # Loop over ALL timesteps (not a chunk)
            for local_t_idx, (t_val, t_shift_val, sigma_val) in enumerate(zip(local_timesteps, local_shift_time, local_sigmas)):
                
                # Get the t value (current) and t_shift (next)
                t_i = torch.tensor([t_val], device=device, dtype=torch.float32)
                t_shift_i = torch.tensor([t_shift_val], device=device, dtype=torch.float32)
                sigma_t = torch.tensor(sigma_val, device=device, dtype=torch.float32)

                # Create the noisy latent x_t from the clean latent x_i
                noise = torch.randn_like(x_i_clean, device=device)
                x_t_i = (1.0 - sigma_t) * x_i_clean + sigma_t * noise

                step_seed = args.base_seed + total_samples_processed_gpu + i + local_t_idx

                try:

                    step_ratio = calculate_solver_step_sensitivity(
                        model_to_test, sample_scheduler, x_t_i, t_i, t_shift_i,
                        y_i, transformer_kwargs_i, device, step_seed
                    )

                    Jt_norm = calculate_jacobian_norm_T(
                        model_to_test, x_t_i, t_i, y_i, t_shift_i, transformer_kwargs_i, device
                    )
                except Exception as e:
                    if is_main_process:
                        logging.warning(f"Error in sensitivity calc at t={t_val:.2f}: {e}. Skipping step.")
                    step_ratio = 0.0
                    Jt_norm = 0.0
                    continue # Skip this timestep


                # Save to the correct *global* index
                global_t_index = local_t_indices_original[local_t_idx]
                
                local_sum_sensitivities[global_t_index, 0] += step_ratio
                local_sum_sensitivities[global_t_index, 1] += Jt_norm
                local_sample_counts[global_t_index] += 1

            samples_processed_in_batch += 1

        total_samples_processed_gpu += samples_processed_in_batch

        # Aggregate and save checkpoint after each batch
        if is_main_process: logging.info("Aggregating and saving checkpoint...")
        aggregate_and_save(
            local_sum_sensitivities, local_sample_counts, local_t_indices_original,
            timesteps_all, device, args, is_main_process, checkpoint=True
        )

    if is_main_process: logging.info("Finished computation. Saving final results.")
    aggregate_and_save(
        local_sum_sensitivities, local_sample_counts, local_t_indices_original,
        timesteps_all, device, args, is_main_process, checkpoint=False
    )

    if dist.is_initialized():
        dist.destroy_process_group()

    if is_main_process: logging.info("Finished.")

# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------
if __name__ == "__main__":
    transformers.utils.logging.set_verbosity_warning()
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    run_sensitivity(args)