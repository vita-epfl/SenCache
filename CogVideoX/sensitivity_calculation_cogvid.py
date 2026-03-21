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
import datetime as dt # Added for DDP Timeout

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from PIL import Image

# --- Diffusers (CogVideoX) Imports ---
from diffusers import CogVideoXPipeline
from diffusers.utils import load_image
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

# --- Wan/SiT Utility Imports ---
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

# --- Sensitivity Imports ---
import torch.backends.cuda
import torch.nn.functional as F_nn
import torch.utils.checkpoint as checkpoint

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------
# Checkpointing Function (from Wan script)
# ---------------------------------------------------------------

def aggregate_and_save(local_sum_sensitivities, local_sample_counts, local_t_indices_original, 
                       timesteps_all, device, args, is_main_process, checkpoint=False):
    """
    Aggregates data from all processes and saves the current means.
    """
    
    # 1. Create full-sized tensors on the device to hold all results
    all_sums_tensor = torch.zeros((args.num_t_steps, 2), device=device, dtype=torch.float32)
    all_counts_tensor = torch.zeros(args.num_t_steps, device=device, dtype=torch.float32)
    
    # 2. Place local results into the correct positions in the full tensors
    local_t_indices_tensor = torch.tensor(local_t_indices_original, device=device, dtype=torch.long)
    all_sums_tensor[local_t_indices_tensor] = torch.tensor(local_sum_sensitivities).to(device)
    all_counts_tensor[local_t_indices_tensor] = torch.tensor(local_sample_counts).to(device)
    
    # 3. Sum results from all processes
    dist.all_reduce(all_sums_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(all_counts_tensor, op=dist.ReduceOp.SUM)

    # 4. Save results (only on main process)
    if is_main_process:
        all_counts_tensor_expanded = all_counts_tensor.unsqueeze(-1)
        current_mean_sensitivities = all_sums_tensor / (all_counts_tensor_expanded + 1e-8)
        current_means_np = current_mean_sensitivities.cpu().numpy()
        
        model_name_safe = args.ckpts_path.replace('/', '_')
        if checkpoint:
            filename = f"sensitivity_checkpoint_{model_name_safe}_t{args.num_t_steps}_s{args.max_samples}.npz"
        else:
            filename = f"sensitivity_results_{model_name_safe}_t{args.num_t_steps}_s{args.max_samples}.npz"
        
        output_path = Path(args.output_dir) / filename
        np.savez(
            output_path, 
            timesteps=timesteps_all, 
            J_x_norm=current_means_np[:, 0], 
            J_t_norm=current_means_np[:, 1],
            sample_counts=all_counts_tensor.cpu().numpy()
        )
        
        save_type = "Checkpoint" if checkpoint else "Final results"
        print(f"\nSaved {save_type} to: {output_path}")

    # 5. Wait for rank 0 to finish saving
    dist.barrier()

# ---------------------------------------------------------------
# Arg Parsing (Merged for CogVideoX)
# ---------------------------------------------------------------

def _validate_args(args):
    assert args.ckpts_path is not None, "Please specify the checkpoint path (--ckpts_path)."
    
    # --- Dataset Args ---
    assert args.json_path is not None, "Please specify --json_path"
    assert args.video_base_path is not None, "Please specify --video_base_path"
    assert os.path.exists(args.json_path), f"JSON file not found at: {args.json_path}"
    assert os.path.exists(args.video_base_path), f"Video base path not found at: {args.video_base_path}"
    
    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate sensitivity for a CogVideoX model from a JSON dataset."
    )
    # --- CogVideoX Args ---
    parser.add_argument('--ckpts_path', type=str, default="THUDM/CogVideoX1.5-5B", help='Path to CogVideoX checkpoint')
    parser.add_argument('--height', type=int, default=480, help='Height of the videos')
    parser.add_argument('--width', type=int, default=720, help='Width of the generated video')
    parser.add_argument('--frame_num', type=int, default=49, help='Number of frames to sample (defaulted to 49).')
    
    # --- Dataset Args (from Wan script) ---
    parser.add_argument("--json_path", type=str, required=True, help="Path to the .json file with video paths and text.")
    parser.add_argument("--video_base_path", type=str, required=True, help="Base directory for video files.")
    parser.add_argument("--num_workers", type=int, default=6, help="Number of workers for the DataLoader.")
    
    # --- Sensitivity Args (from Wan script) ---
    parser.add_argument("--num_t_steps", type=int, default=50, help="Number of timesteps to evaluate (and for inference).")
    parser.add_argument("--max_samples", type=int, default=2048, help="Total number of samples to process from the dataset.")
    parser.add_argument("--output_dir", type=str, default="./results_sensitivity_cogvideox", help="Directory to save the .npz file.")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of samples per GPU before aggregating (DataLoader batch size).")
    parser.add_argument("--base_seed", type=int, default=42, help="The seed to use for sampling.")

    args = parser.parse_args()
    _validate_args(args)
    return args

# ---------------------------------------------------------------
# Logging (from Wan script)
# ---------------------------------------------------------------

def _init_logging(rank):
    if rank == 0:
        logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s", handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)
    
    # Silence transformers and PIL
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.getLogger('PIL').setLevel(logging.ERROR)

# ---------------------------------------------------------------
# New Dataset Class (Adapted for CogVideoX)
# ---------------------------------------------------------------

class VideoTextJSONDataset(Dataset):
    """
    Loads videos and text from the provided JSON file.
    Performs video loading and transformation on the CPU worker.
    MODIFIED: Returns (video_tensor, text_string)
    """
    def __init__(self, json_path, video_base_path, size, frame_num):
        super().__init__()
        try:
            self.data = json.load(open(json_path))
        except Exception as e:
            raise ValueError(f"Error loading JSON file {json_path}: {e}")
            
        self.base_path = Path(video_base_path)
        self.frame_num = frame_num
        
        # H, W
        if isinstance(size, tuple):
             # size is (H, W)
             pass
        elif isinstance(size, str):
            H, W = map(int, size.split('*'))
            size = (H, W)
        else:
            raise ValueError(f"Invalid size format: {size}")
            
        # We need to resize each frame.
        self.transform = transforms.Compose([
            transforms.Resize(size, antialias=True)
        ])
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        try:
            item = self.data[idx]
            
            # --- MODIFIED: Load text ---
            try:
                text = item['text'] 
            except KeyError:
                logging.warning(f"Item at index {idx} has no 'text' field. Skipping.")
                return self.__getitem__((idx + 1) % len(self))
                
            file_path_key = item['file_path']
            video_path = self.base_path / file_path_key
        
            # 1. Load video with decord
            vr = VideoReader(str(video_path), ctx=cpu(0))
            total_frames = len(vr)
            
            # 2. Sample frames
            indices = np.linspace(0, total_frames - 1, self.frame_num, dtype=int)
            frames = vr.get_batch(indices).asnumpy() # (F, H, W, C)
            
            # 3. Transform to (F, C, H, W) for torchvision
            frames_th = torch.tensor(frames).permute(0, 3, 1, 2) # (F, C, H, W)
            
            # 4. Resize and Normalize
            frames_resized = self.transform(frames_th) # (F, C, H_new, W_new)
            video_tensor = (frames_resized.float() / 255.0) * 2.0 - 1.0 # Normalize [-1, 1]
            
            # 5. Permute to (C, F, H, W) for the model (standard VAE input)
            video_tensor = video_tensor.permute(1, 0, 2, 3) # (C, F, H, W)
            
            # --- RETURN VIDEO AND TEXT ---
            return video_tensor, text
            
        except Exception as e:
            logging.warning(f"Rank {dist.get_rank() if dist.is_initialized() else 0}: Error loading video {video_path}: {e}. Skipping.")
            # On error, return the next item
            return self.__getitem__((idx + 1) % len(self))



def calculate_solver_step_sensitivity(
    model,              # This is pipe.transformer
    sample_scheduler,   # This is pipe.scheduler
    x_t,                # The (F, C, H, W) latent at time t
    t,                  # The (1,) tensor for time t
    t_next,             # The (1,) tensor for time t-1
    y,                  # The conditioning dict {'encoder_hidden_states': ...}
    device,
    generator_seed      # A seed for the sampler step
):
    """
    Calculates the ratio of change in model output vs. change in model input
    across a SINGLE, FULL sampler step.
    
    This measures: ||f(x_{t-1}, t) - f(x_t, t)|| / ||x_{t-1} - x_t||
    """
    
    g = torch.Generator(device=device).manual_seed(generator_seed)
    
    def model_fn(x_in_tensor, t_in_tensor):
        """
        A helper function to correctly call the diffusers transformer.
        Takes a (F, C, H, W) tensor and returns a (F, C, H, W) tensor.
        """
        

        x_in_tensor_f32 = x_in_tensor.to(torch.float32)


        # The model's forward() expects a batch dimension
        x_in_batch = x_in_tensor_f32.unsqueeze(0) # Now (1, F, C, H, W)
        
        # Transformer expects integer timesteps
        t_in_int = t_in_tensor.to(torch.int)
        
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # Call the model
            pred_sample = model(
                hidden_states=x_in_batch,
                timestep=t_in_int,
                **y  # This passes {'encoder_hidden_states': ...}
            ).sample
            
        # Cast output to float32 to avoid mixing with x_t in scheduler
        return pred_sample.squeeze(0).to(torch.float32)


    pred_t = model_fn(x_t, t)
    

    pred_t_batch = pred_t.unsqueeze(0)  # (1, F, C, H, W), float32
    x_t_batch = x_t.unsqueeze(0)        # (1, F, C, H, W), float32
    

    x_t_next_batch = sample_scheduler.step(
        pred_t_batch,
        t.to(torch.int), 
        x_t_batch,
        return_dict=False,
        generator=g
    )[0]
    
    x_t_next = x_t_next_batch.squeeze(0) # (F, C, H, W), possibly float64


    pred_t_next = model_fn(x_t_next, t) # Output is float32
    

    with torch.no_grad():
        delta_x_norm = torch.norm(x_t_next - x_t).item()
        delta_pred_norm = torch.norm(pred_t_next - pred_t).item()
    

    if delta_x_norm < 1e-8:
        return 0.0
    
    return delta_pred_norm / delta_x_norm


def calculate_jacobian_norm_T(model, x, t, y, t_next, device):
    """
    Calculates || (f(x_t, t_next) - f(x_t, t)) / (t_next - t) ||
    """
    x_batch = x.unsqueeze(0).to(device, dtype=torch.float32) # (1, F, C, H, W)
    t = t.to(device, dtype=torch.float32)
    t_next = t_next.to(device, dtype=torch.float32)

    def model_fn_t_simple(t_in):
        """ Helper function for forward pass with a specific time """
        t_in_int = t_in.to(torch.int)
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = model(
                hidden_states=x_batch, 
                timestep=t_in_int, 
                **y
            ).sample
        
        # Cast output to float32 to avoid mixing dtypes
        return output.to(torch.float32) # Returns (1, F, C, H, W)


    with torch.no_grad():
        f_t_gpu = model_fn_t_simple(t)
        f_t_next_gpu = model_fn_t_simple(t_next)
        
        time_delta = t_next.item() - t.item()
        if abs(time_delta) < 1e-8:
             return 0.0 
             
        J_t_vector = (f_t_next_gpu - f_t_gpu) / time_delta
        del f_t_next_gpu, f_t_gpu
    
    return J_t_vector.norm().item()




def run_sensitivity(args):
    """The main worker function for each GPU process."""
    
    # 1. --- DDP Setup ---
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)
    is_main_process = (rank == 0)

    ddp_timeout = dt.timedelta(minutes=120)
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl", 
        init_method="env://", 
        rank=rank, 
        world_size=world_size,
        timeout=ddp_timeout
    )

    # Broadcast seed
    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]
    torch.manual_seed(args.base_seed + rank)
    np.random.seed(args.base_seed + rank)
    random.seed(args.base_seed + rank)

    # 2. --- Load Model (CogVideoX) ---
    if is_main_process: 
        logging.info(f"Analysis job args: {args}")
        logging.info(f"Creating CogVideoX pipeline from: {args.ckpts_path}")
    
    Image.MAX_IMAGE_PIXELS = None 
    transformers.utils.logging.set_verbosity_error()
    
    pipe = CogVideoXPipeline.from_pretrained(args.ckpts_path, torch_dtype=torch.bfloat16)
    pipe.to(device)
    # Enable VAE slicing for memory efficiency
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    model_to_test = pipe.transformer
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    sample_scheduler = pipe.scheduler
    
    patch_size_t = pipe.transformer.config.patch_size_t
    if patch_size_t is None:
        patch_size_t = 1 
    if is_main_process:
        logging.info(f"Using temporal patch size (patch_size_t): {patch_size_t}")


    # 3. --- Prepare Data (NEW) ---
    size_tuple = (args.height, args.width)

    if is_main_process:
        logging.info(f"Target video size: {size_tuple}, Frames: {args.frame_num}")
        
    dataset = VideoTextJSONDataset(
        json_path=args.json_path,
        video_base_path=args.video_base_path,
        size=size_tuple,
        frame_num=args.frame_num
    )
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.base_seed
    )
    
    if args.batch_size % world_size != 0:
        logging.warning(f"Batch size ({args.batch_size}) is not divisible by world size ({world_size}).")
        
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


    # 4. --- Prepare Timesteps (Scheduler) ---
    sample_scheduler.set_timesteps(args.num_t_steps, device=device)
    timesteps_tensor = sample_scheduler.timesteps
    
    # Ensure they are sorted descending
    timesteps_tensor, _ = torch.sort(timesteps_tensor, descending=True)


    if len(timesteps_tensor) > args.num_t_steps:
        if is_main_process:
            logging.info(f"Scheduler returned {len(timesteps_tensor)} steps. Truncating to {args.num_t_steps}.")
        timesteps_tensor = timesteps_tensor[:args.num_t_steps]
    
    timesteps_all = timesteps_tensor.cpu().numpy()
    
    if is_main_process:
        logging.info(f"Using {sample_scheduler.__class__.__name__} scheduler.")
        logging.info(f"Timestep range: [{timesteps_all.min():.0f}, {timesteps_all.max():.0f}] with {len(timesteps_all)} steps.")
    

    # t_next values (t_{i-1})
    shift_time = np.roll(timesteps_all, -1)
    shift_time[-1] = 0 # Last step shifts to 0
    

    local_timesteps = timesteps_all
    local_shift_time = shift_time
    

    local_t_indices_original = list(range(len(timesteps_all)))


    local_sum_sensitivities = np.zeros((len(timesteps_all), 2), dtype=np.float32)
    local_sample_counts = np.zeros(len(timesteps_all), dtype=np.float32) 
    total_samples_processed_gpu = 0 
    data_iter = iter(loader)

    if is_main_process:
        logging.info(f"Starting sensitivity calculation on {world_size} GPUs.")
        logging.info(f"Total samples to process: {args.max_samples}, Max samples per GPU: {max_samples_per_gpu}")
        logging.info(f"Each GPU will process ALL {len(local_timesteps)} timesteps for each sample.")
    

    while total_samples_processed_gpu < max_samples_per_gpu:
        try:
            video_batch, text_batch = next(data_iter)
        except StopIteration:
            if is_main_process: logging.info("Restarting data iterator...")
            sampler.set_epoch(sampler.epoch + 1)
            data_iter = iter(loader)
            try:
                video_batch, text_batch = next(data_iter)
            except StopIteration:
                if is_main_process: logging.warning("DataLoader empty even after reset. Ending.")
                break 

        video_batch = video_batch.to(device)
        
        with torch.no_grad():
            # 1. Encode text
            prompt_inputs = tokenizer(
                list(text_batch), 
                padding="max_length", 
                max_length=tokenizer.model_max_length, 
                truncation=True, 
                return_tensors="pt"
            )
            prompt_embeds_batch = text_encoder(prompt_inputs.input_ids.to(device))[0].to(torch.bfloat16)

            # 2. Encode video
            # VAE Input: (B, C, F, H, W) -> Output: (B, C_latent, F_latent, H_latent, W_latent)
            latents_batch_vae_shape = vae.encode(video_batch.to(torch.bfloat16)).latent_dist.sample()
            
            # Cast scaling_factor to float32 for stable math
            scaling_factor = torch.tensor(vae.config.scaling_factor, dtype=torch.float32, device=device)
            scaled_latents = latents_batch_vae_shape.to(torch.float32) * scaling_factor

            # Permute from (B, C, F, H, W) to (B, F, C, H, W) for the transformer
            latents_batch = scaled_latents.permute(0, 2, 1, 3, 4).contiguous()
            
            # Pad temporal dimension to be multiple of patch_size_t
            b, f, c, h, w = latents_batch.shape
            if f % patch_size_t != 0:
                pad_frames = patch_size_t - (f % patch_size_t)
                # Pad (0, 0) for W, (0, 0) for H, (0, 0) for C, (0, pad_frames) for F
                latents_batch = F_nn.pad(latents_batch, (0, 0, 0, 0, 0, 0, 0, pad_frames), "constant", 0)
            

        batch_loop = tqdm(
            range(len(latents_batch)), 
            desc=f"Rank {rank} (Processed: {total_samples_processed_gpu})",
            disable=(not is_main_process)
        )
        
        samples_processed_in_batch = 0
        for i in batch_loop:
            if (total_samples_processed_gpu + samples_processed_in_batch) >= max_samples_per_gpu:
                break
                
            x_i = latents_batch[i] # Clean latent (F_padded, C, H, W), float32
            context_i = prompt_embeds_batch[i].unsqueeze(0) 
            y_i = {'encoder_hidden_states': context_i}

            # --- Loop over ALL timesteps (not a chunk) ---
            for local_t_idx, (t_val, t_shift_val) in enumerate(zip(local_timesteps, local_shift_time)):
                
                t_i = torch.tensor([t_val], device=device, dtype=torch.float32)
                t_shift_i = torch.tensor([t_shift_val], device=device, dtype=torch.float32)
                
                # Add noise using the scheduler's method
                noise = torch.randn_like(x_i, device=device) # float32
                x_t_i = sample_scheduler.add_noise(x_i.unsqueeze(0), noise.unsqueeze(0), t_i.to(torch.int)).squeeze(0) # float32

                try:
                    step_seed = (args.base_seed + total_samples_processed_gpu + i + local_t_idx)
                    
                    # J_x_norm
                    step_ratio = calculate_solver_step_sensitivity(
                        model_to_test,
                        sample_scheduler,
                        x_t_i,
                        t_i,
                        t_shift_i,
                        y_i,
                        device,
                        step_seed
                    )
                    
                    # J_t_norm
                    Jt_norm = calculate_jacobian_norm_T(
                        model_to_test, x_t_i, t_i, y_i, t_shift_i, device
                    )
                except Exception as e:
                    if is_main_process: 
                        logging.warning(f"Warning: Error in sensitivity calculation: {e}. Skipping step.")
                    torch.cuda.empty_cache()
                    continue
                

                # Save to the correct *global* index
                global_t_index = local_t_indices_original[local_t_idx]

                local_sum_sensitivities[global_t_index, 0] += step_ratio
                local_sum_sensitivities[global_t_index, 1] += Jt_norm
                local_sample_counts[global_t_index] += 1
            
            samples_processed_in_batch += 1
        
        total_samples_processed_gpu += samples_processed_in_batch

        # 7. --- Aggregate and Save Checkpoint ---
        if is_main_process:
            total_global_samples = total_samples_processed_gpu * world_size
            logging.info(f"\nAggregating and saving checkpoint at ~{total_global_samples} total samples...")
            
        aggregate_and_save(
            local_sum_sensitivities, local_sample_counts, local_t_indices_original,
            timesteps_all, device, args, is_main_process, 
            checkpoint=True
        )
    
    # 8. --- Final Save & Cleanup ---
    if is_main_process: 
        logging.info(f"\nRank {rank}: Finished computation. Saving final results.")

    aggregate_and_save(
        local_sum_sensitivities, local_sample_counts, local_t_indices_original,
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
    transformers.utils.logging.set_verbosity_warning()
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    run_sensitivity(args)