## ⚙️ Installation & Setup

1. **Clone the Base Repository:** First, follow the official instructions in the [Wan2.1 repository](https://github.com/Wan-Video/Wan2.1) to clone their code and complete the environment installation.
2. **Download Weights:** Download the sensitivity weights and place them directly into your cloned Wan2.1 repository.
3. **Add the Script:** Copy the `sencache.py` file from this repository and paste it into the root directory of your cloned Wan2.1 repository.

## 🚀 Usage

Navigate to your Wan2.1 directory where you placed the script and weights. 

To run Text-to-Video (T2V) generation using the **1.3B model**, use the following command:

```bash
python sencache.py \
  --ckpt_dir ./Wan2.1-T2V-1.3B \
  --task t2v-1.3B \
  --size 832*480 \
  --output_dir ./output \
  --offload_model True \
  --prompt_file ./test_prompt.json \
  --frame_num 81 \
  --sample_steps 50 \
  --sencache_K 3 \
  --sencache_thresh_main 2 \
  --sencache_thresh_start 0.045
```

### **Key Arguments:**
* `--prompt_file`: Path to your JSON file containing generation prompts.
* `--sencache_K`: The step interval for SenCache updates.
* `--sencache_thresh_main`: The main threshold value for caching.
* `--sencache_thresh_start`: The starting threshold value for caching.

---

## 🙏 Acknowledgements

We would like to thank the contributors to [Wan2.1](https://github.com/Wan-Video/Wan2.1) for their foundational work and models.

