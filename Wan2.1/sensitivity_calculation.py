import argparse
import logging
import os
import sys
import warnings
from datetime import datetime
import random
from pathlib import Path
import json
import math
import datetime as dt 

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from PIL import Image

# --- New Imports for Data Loading ---
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms


from wan.utils.fm_solvers import (FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler as FlowUniPCMultistepSchedulerAlt
try:
    from decord import VideoReader, cpu
except ImportError:
    print("Error: 'decord' library not found. Please install it with 'pip install decord'")
    sys.exit(1)

# --- Wan Imports ---
import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.utils import str2bool

# --- Sensitivity Imports ---
import torch.backends.cuda
import torch.nn.functional as F_nn
import torch.utils.checkpoint as checkpoint

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------
# Checkpointing Function (UPDATED FOR EXACT VALUES)
# ---------------------------------------------------------------

def gather_and_save_exact(local_raw_data, total_samples_processed_gpu, 
                          timesteps_all, device, args, is_main_process, checkpoint=False):
    """
    Gathers raw data from all processes and saves the exact values.
    """

    local_tensor = torch.tensor(local_raw_data, device=device, dtype=torch.float32)
    local_count = torch.tensor([total_samples_processed_gpu], device=device, dtype=torch.long)

    # Prepare lists for gathering on Rank 0
    if is_main_process:
        gathered_data = [torch.zeros_like(local_tensor) for _ in range(dist.get_world_size())]
        gathered_counts = [torch.zeros_like(local_count) for _ in range(dist.get_world_size())]
    else:
        gathered_data = None
        gathered_counts = None

    # Gather data
    dist.gather(local_tensor, gather_list=gathered_data, dst=0)
    dist.gather(local_count, gather_list=gathered_counts, dst=0)

    # Save results (only on main process)
    if is_main_process:
        all_valid_data = []
        
        # Iterate through gathered data and slice based on valid counts
        for i, (data_tensor, count_tensor) in enumerate(zip(gathered_data, gathered_counts)):
            valid_samples = count_tensor.item()
            if valid_samples > 0:
                # Slice: Take only valid rows [0 : valid_samples]
                # Shape becomes (valid_samples, num_t_steps, 2)
                all_valid_data.append(data_tensor[:valid_samples].cpu().numpy())

        # Concatenate along the sample dimension (axis 0)
        if len(all_valid_data) > 0:
            full_dataset = np.concatenate(all_valid_data, axis=0)
        else:
            full_dataset = np.empty((0, args.num_t_steps, 2))

        model_name_safe = args.task.replace('/', '_')
        if checkpoint:
            filename = f"raw_sensitivity_checkpoint_{model_name_safe}_t{args.num_t_steps}_s{args.max_samples}.npz"
        else:
            filename = f"raw_sensitivity_results_{model_name_safe}_t{args.num_t_steps}_s{args.max_samples}.npz"
        
        output_path = Path(args.output_dir) / filename
        
        # Save exact values
        # raw_data shape: (Total_Samples, Num_Timesteps, 2)
        #   [:, :, 0] -> Step Sensitivity
        #   [:, :, 1] -> Time Sensitivity
        np.savez(
            output_path, 
            timesteps=timesteps_all, 
            raw_data=full_dataset
        )
        
        save_type = "Checkpoint" if checkpoint else "Final results"
        print(f"\nSaved {save_type} (Exact Values) to: {output_path}")
        print(f"Shape of saved data: {full_dataset.shape}")

    # Wait for rank 0 to finish saving
    dist.barrier()

# ---------------------------------------------------------------
# Arg Parsing
# ---------------------------------------------------------------

def _validate_args(args):
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert "t2v" in args.task, f"This script is adapted for 't2v' tasks only, not {args.task}"

    if args.frame_num is None:
        args.frame_num = 81

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)
    assert args.size in SUPPORTED_SIZES[args.task], f"Unsupported size {args.size} for task {args.task}"
    
    assert args.json_path is not None, "Please specify --json_path"
    assert args.video_base_path is not None, "Please specify --video_base_path"
    assert os.path.exists(args.json_path), f"JSON file not found at: {args.json_path}"
    assert os.path.exists(args.video_base_path), f"Video base path not found at: {args.video_base_path}"
    
    assert args.embedding_cache_path is not None, "Please specify --embedding_cache_path"
    assert os.path.exists(args.embedding_cache_path), f"Embedding cache file not found at: {args.embedding_cache_path}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate sensitivity for a Wan text-to-video model from a JSON dataset."
    )
    # --- Wan Args ---
    parser.add_argument("--task", type=str, default="t2v-14B", help="The t2v task to run.")
    parser.add_argument("--size", type=str, default="832*480", help="The area (width*height) of the videos.")
    parser.add_argument("--frame_num", type=int, default=None, help="How many frames to sample. Default 81 for t2v.")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="The path to the checkpoint directory.")
    parser.add_argument("--offload_model", type=str2bool, default=None, help="Whether to offload the model to CPU.")
    parser.add_argument("--ulysses_size", type=int, default=1, help="Ulysses parallelism size.")
    parser.add_argument("--ring_size", type=int, default=1, help="Ring attention parallelism size.")
    parser.add_argument("--t5_fsdp", action="store_true", default=False, help="Use FSDP for T5.")
    parser.add_argument("--t5_cpu", action="store_true", default=False, help="Place T5 model on CPU.")
    parser.add_argument("--dit_fsdp", action="store_true", default=False, help="Use FSDP for DiT.")
    parser.add_argument("--base_seed", type=int, default=42, help="The seed to use for sampling.")
    
    # --- Dataset Args ---
    parser.add_argument("--json_path", type=str, required=True, help="Path to the .json file with video paths and text.")
    parser.add_argument("--video_base_path", type=str, required=True, help="Base directory for video files.")
    parser.add_argument("--num_workers", type=int, default=6, help="Number of workers for the DataLoader.")
    parser.add_argument("--embedding_cache_path", type=str, required=True, help="Path to the pre-computed embedding cache .pt file.")
    
    # --- Sensitivity Args ---
    parser.add_argument("--num_t_steps", type=int, default=50, help="Number of timesteps to evaluate.")
    parser.add_argument("--num_power_iter", type=int, default=5, help="Power iterations for spectral norm.")
    parser.add_argument("--max_samples", type=int, default=2048, help="Total number of samples to process from the dataset.")
    parser.add_argument("--output_dir", type=str, default="./results_sensitivity_wan", help="Directory to save the .npz file.")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of samples per GPU before aggregating and saving (DataLoader batch size).")

    parser.add_argument("--sample_solver", type=str, default='unipc', choices=['unipc', 'dpm++'], help="The solver used to sample timesteps.")
    parser.add_argument("--sample_shift", type=float, default=5.0, help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument("--num_train_timesteps", type=int, default=1000, help="Number of training timesteps (should match model training config).")
    
    args = parser.parse_args()
    _validate_args(args)
    return args

# ---------------------------------------------------------------
# Logging 
# ---------------------------------------------------------------

def _init_logging(rank):
    if rank == 0:
        logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s", handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)

# ---------------------------------------------------------------
# Dataset Class
# ---------------------------------------------------------------

class VideoTextJSONDataset(Dataset):
    def __init__(self, json_path, video_base_path, size, frame_num):
        super().__init__()
        try:
            self.data = json.load(open(json_path))
        except Exception as e:
            raise ValueError(f"Error loading JSON file {json_path}: {e}")
            
        self.base_path = Path(video_base_path)
        self.frame_num = frame_num
        
        if isinstance(size, str):
            H, W = map(int, size.split('*'))
            size = (H, W)
            
        self.transform = transforms.Compose([
            transforms.Resize(size, antialias=True)
        ])
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        file_path_key = item['file_path']
        video_path = self.base_path / file_path_key
        
        try:
            vr = VideoReader(str(video_path), ctx=cpu(0))
            total_frames = len(vr)
            indices = np.linspace(0, total_frames - 1, self.frame_num, dtype=int)
            frames = vr.get_batch(indices).asnumpy()
            frames_th = torch.tensor(frames).permute(0, 3, 1, 2)
            frames_resized = self.transform(frames_th)
            video_tensor = (frames_resized.float() / 255.0) * 2.0 - 1.0
            video_tensor = video_tensor.permute(1, 0, 2, 3)
            return video_tensor, file_path_key
            
        except Exception as e:
            logging.warning(f"Rank {dist.get_rank() if dist.is_initialized() else 0}: Error loading video {video_path}: {e}. Skipping.")
            return self.__getitem__((idx + 1) % len(self))

# ---------------------------------------------------------------
# Sensitivity computation
# ---------------------------------------------------------------

@torch.no_grad()
def compute_alpha_sigma(t):
    alpha = 1-t
    sigma = t
    return alpha, sigma

def calculate_solver_step_sensitivity(
    model, sample_scheduler, x_t, t, t_next, y, device, generator_seed
):
    g = torch.Generator(device=device).manual_seed(generator_seed)
    
    def model_fn(x_in_tensor, t_in_tensor):
        x_in_list = [x_in_tensor] 
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            pred_list = model(x_in_list, t=t_in_tensor, **y)
            return pred_list[0] 

    pred_t = model_fn(x_t, t)
    pred_t_batch = pred_t.unsqueeze(0)
    x_t_batch = x_t.unsqueeze(0) 
    
    x_t_next_batch = sample_scheduler.step(
        pred_t_batch, t, x_t_batch, return_dict=False, generator=g
    )[0]
    
    x_t_next = x_t_next_batch.squeeze(0)
    pred_t_next = model_fn(x_t_next, t) 
    
    with torch.no_grad():
        delta_x_norm = torch.norm(x_t_next - x_t).item()
        delta_pred_norm = torch.norm(pred_t_next - pred_t).item()
    
    if delta_x_norm < 1e-8:
        return 0.0
    return delta_pred_norm / delta_x_norm

def calculate_jacobian_norm_T(model, x, t, y, t_shift, device):
    x = x.unsqueeze(0).to(device, dtype=torch.float32)
    
    epsilon = t_shift
    t = t.to(device, dtype=torch.float32)
    epsilon = epsilon.to(device, dtype=torch.float32)

    def model_fn_t_simple(t_in):
        output = model(x, t=t_in, **y)[0] 
        return output

    with torch.no_grad():
        f_t_gpu = model_fn_t_simple(t)
        f_t_plus_eps_gpu = model_fn_t_simple(epsilon)
        J_t_vector = (f_t_plus_eps_gpu - f_t_gpu) / (epsilon - t)

        del f_t_plus_eps_gpu
        del f_t_gpu
            
    return J_t_vector.norm().item()


# ---------------------------------------------------------------
# Main sensitivity analysis
# ---------------------------------------------------------------

def run_sensitivity(args):
    """The main worker function for each GPU process."""
    
    # 1. --- DDP Setup ---
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)
    is_main_process = (rank == 0)

    if args.offload_model is None:
        args.offload_model = False
    
    ddp_timeout = dt.timedelta(minutes=120)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl", init_method="env://", rank=rank, world_size=world_size, timeout=ddp_timeout
    )

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]
    torch.manual_seed(args.base_seed + rank)
    np.random.seed(args.base_seed + rank)
    random.seed(args.base_seed + rank)

    # 2. --- Load Model ---
    cfg = WAN_CONFIGS[args.task]
    if is_main_process: 
        logging.info(f"Analysis job args: {args}")
        logging.info("Creating WanT2V pipeline for sensitivity analysis.")
    
    wan_t2v = wan.WanT2V(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=device, rank=rank,
        t5_fsdp=args.t5_fsdp, dit_fsdp=args.dit_fsdp,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu,
    )

    model_to_test = wan_t2v.model
    vae = wan_t2v.vae

    if is_main_process:
        logging.info(f"Loading embedding cache from {args.embedding_cache_path}...")
    
    embedding_cache_map = torch.load(args.embedding_cache_path, map_location='cpu')
    
    if is_main_process:
        logging.info(f"Loaded {len(embedding_cache_map)} cached embeddings.")


    # 3. --- Prepare Data ---
    H, W = SIZE_CONFIGS[args.size]
    size_tuple = (H, W)

    F = args.frame_num
    target_shape = (wan_t2v.vae.model.z_dim, (F - 1) // wan_t2v.vae_stride[0] + 1,
                    size_tuple[0] // wan_t2v.vae_stride[1],
                    size_tuple[1] // wan_t2v.vae_stride[2])

    seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                        (wan_t2v.patch_size[1] * wan_t2v.patch_size[2]) *
                        target_shape[1] / wan_t2v.sp_size) * wan_t2v.sp_size
    
    dataset = VideoTextJSONDataset(
        json_path=args.json_path,
        video_base_path=args.video_base_path,
        size=size_tuple,
        frame_num=args.frame_num
    )
    
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.base_seed
    )
    
    # Calculate samples per GPU
    max_samples_per_gpu = math.ceil(args.max_samples / world_size)
    
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True
    )
    
    if is_main_process:
        logging.info(f"Dataset size: {len(dataset)}. Sampler size per GPU: {len(sampler)}")
        logging.info(f"Total max samples: {args.max_samples}. Max samples per GPU: {max_samples_per_gpu}")


    # 4. --- Prepare Timesteps ---
    if args.sample_solver == 'unipc':
        sample_scheduler = FlowUniPCMultistepSchedulerAlt(
            num_train_timesteps=args.num_train_timesteps, shift=1, use_dynamic_shifting=False
        )
        dummy_scheduler = FlowUniPCMultistepSchedulerAlt(
            num_train_timesteps=args.num_train_timesteps, shift=1, use_dynamic_shifting=False
        )
        dummy_scheduler.set_timesteps(args.num_t_steps, device='cpu', shift=args.sample_shift)
        timesteps_tensor = dummy_scheduler.timesteps
        sigmas_tensor = dummy_scheduler.sigmas
        
    elif args.sample_solver == 'dpm++':
        from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler
        sample_scheduler = FlowDPMSolverMultistepScheduler(
            num_train_timesteps=args.num_train_timesteps, shift=1, use_dynamic_shifting=False
        )
        sampling_sigmas = get_sampling_sigmas(args.num_t_steps, args.sample_shift)
        timesteps_tensor, _ = retrieve_timesteps(sample_scheduler, device='cpu', sigmas=sampling_sigmas)
        sigmas_tensor = sampling_sigmas
    else:
        raise NotImplementedError("Unsupported solver.")
    

    timesteps_all = timesteps_tensor.cpu().numpy()
    sigmas_all = sigmas_tensor.cpu().numpy()
    
    shift_time = np.roll(timesteps_all, -1)
    shift_time[-1] = 0.0

    local_timesteps = timesteps_all
    local_sigmas = sigmas_all
    local_shift_time = shift_time

    # 5. --- Initialize Storage for EXACT VALUES ---
    local_raw_data = np.zeros((max_samples_per_gpu, args.num_t_steps, 2), dtype=np.float32)
    
    total_samples_processed_gpu = 0 
    data_iter = iter(loader)

    if is_main_process:
        logging.info(f"Starting sensitivity calculation (EXACT VALUES MODE).")
    
    # 6. --- Main Loop ---
    while total_samples_processed_gpu < max_samples_per_gpu:
        try:
            video_batch, path_batch = next(data_iter)
        except StopIteration:
            if is_main_process: logging.info("Restarting data iterator...")
            sampler.set_epoch(sampler.epoch + 1)
            data_iter = iter(loader)
            try:
                video_batch, path_batch = next(data_iter)
            except StopIteration:
                break 

        video_batch = video_batch.to(device)

        with torch.no_grad():
            try:
                context_batch = [embedding_cache_map[path] for path in path_batch]
            except KeyError as e:
                logging.error(f"Rank {rank}: Key error {e}. Skipping batch.")
                continue
            latents_batch = vae.encode(video_batch)

        batch_loop = tqdm(
            range(len(latents_batch)), 
            desc=f"Rank {rank} (Processed: {total_samples_processed_gpu})",
            disable=(not is_main_process)
        )
        
        samples_processed_in_batch = 0
        for i in batch_loop:
            if (total_samples_processed_gpu + samples_processed_in_batch) >= max_samples_per_gpu:
                break
                
            x_i = latents_batch[i] 
            context_i = [context_batch[i].to(device, dtype=torch.float32)]
            y_i = {'context': context_i, 'seq_len': seq_len}

            if args.sample_solver == 'unipc':
                sample_scheduler.set_timesteps(args.num_t_steps, device=device, shift=args.sample_shift)
            elif args.sample_solver == 'dpm++':
                sampling_sigmas = get_sampling_sigmas(args.num_t_steps, args.sample_shift)
                timesteps_tensor, _ = retrieve_timesteps(sample_scheduler, device=device, sigmas=sampling_sigmas)

            # --- Loop over timesteps ---
            for local_t_idx, (t_val, sigma_val, t_shift) in enumerate(zip(local_timesteps, local_sigmas, local_shift_time)):
                
                t_i = torch.tensor([t_val], device=device, dtype=torch.float32)
                sigma_t = torch.tensor(sigma_val, device=device, dtype=torch.float32)
                t_shift = torch.tensor([t_shift], device=device, dtype=torch.float32)
                
                noise = torch.randn_like(x_i, device=device)
                x_t_i = (1.0 - sigma_t) * x_i + sigma_t * noise

                try:
                    step_seed = (args.base_seed + total_samples_processed_gpu + i + local_t_idx)
        
                    step_ratio = calculate_solver_step_sensitivity(
                        model_to_test, sample_scheduler, x_t_i, t_i, t_shift, y_i, device, step_seed
                    )
                    Jt_norm = calculate_jacobian_norm_T(
                        model_to_test, x_t_i, t_i, y_i, t_shift, device
                    )
                except Exception as e:
                    if is_main_process: 
                        logging.warning(f"Error at t={t_val:.4f}: {e}. Skipping step.")
                    continue
                
                # --- STORE EXACT VALUE ---
                # Store in the pre-allocated array at [sample_index, timestep_index, metric_index]
                current_sample_idx = total_samples_processed_gpu + samples_processed_in_batch
                local_raw_data[current_sample_idx, local_t_idx, 0] = step_ratio
                local_raw_data[current_sample_idx, local_t_idx, 1] = Jt_norm
            
            samples_processed_in_batch += 1
        
        total_samples_processed_gpu += samples_processed_in_batch

        # 7. --- Aggregate and Save Checkpoint (Exact Values) ---
        if is_main_process:
            total_global_samples = total_samples_processed_gpu * world_size
            logging.info(f"\nGathering and saving checkpoint at ~{total_global_samples} total samples...")
            
        gather_and_save_exact(
            local_raw_data, total_samples_processed_gpu,
            timesteps_all, device, args, is_main_process, 
            checkpoint=True
        )
    
    # 8. --- Final Save ---
    if is_main_process: 
        logging.info(f"\nRank {rank}: Finished computation. Saving final results.")

    gather_and_save_exact(
        local_raw_data, total_samples_processed_gpu,
        timesteps_all, device, args, is_main_process, 
        checkpoint=False 
    )
    
    if dist.is_initialized():
        dist.destroy_process_group()
    
    if is_main_process:
        logging.info("Finished.")


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    run_sensitivity(args)