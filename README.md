# 💠 SenCache: Accelerating Diffusion Model Inference via Sensitivity-Aware Caching

Yasaman Haghighi, Alexandre Alahi
 
École Polytechnique Fédérale de Lausanne (EPFL)

<p align="center">
  <a href="https://arxiv.org/abs/2602.24208"><b>📄 Paper</b></a> •
  <a href="https://vita-epfl.github.io/SenCache.io/"><b>🌐 Website</b></a>
</p>

---
> 🧠 **Abstract:**
> Diffusion models achieve state-of-the-art video generation quality, but their inference remains expensive due to the large number of sequential denoising steps. This has motivated a growing line of
> research on accelerating diffusion inference. Among training-free acceleration methods, caching reduces computation by reusing previously computed model outputs across timesteps. Existing caching methods
> rely on heuristic criteria to choose cache/reuse timesteps and require extensive tuning. We address this limitation with a principled sensitivity-aware caching framework. Specifically, we formalize the
> caching error through an analysis of the model output sensitivity to perturbations in the denoising inputs, i.e., the noisy latent and the timestep, and show that this sensitivity is a key predictor of
> caching error. Based on this analysis, we propose Sensitivity-Aware Caching **SenCache**, a dynamic caching policy that adaptively selects caching timesteps on a per-sample basis. Our framework provides a
> theoretical basis for adaptive caching, explains why prior empirical heuristics can be partially effective, and extends them to a dynamic, sample-specific approach. Experiments on Wan 2.1, CogVideoX,
> and LTX-Video show that SenCache achieves better visual quality than existing caching methods under similar computational budgets.
---
#### News

- **[Integration]** SenCache has been integrated to accelerate Wan2.2 in [MaxDiffusion](https://github.com/AI-Hypercomputer/maxdiffusion).
- **[CVPR 2026]** SenCache has been accepted for **oral presentation** at CVPR 2026! 🎉

---

#### 🔽 Precomputed Sensitivity weights

We provide precomputed sensitivity weights on Hugging Face:

👉 [Sensitivities](https://huggingface.co/datasets/Yassaman/SenCache)

---

## 🙏 Acknowledgement

This repository is built upon the foundational contributions of **[TeaCache](https://github.com/ali-vilab/TeaCache)**, **[MagCache](https://github.com/Zehong-Ma/MagCache)**, **[Diffusers](https://github.com/huggingface/diffusers)**, **[Wan2.1](https://github.com/Wan-Video/Wan2.1)**, **[CogVideoX](https://github.com/THUDM/CogVideo)** and **[LTX-Video](https://github.com/Lightricks/LTX-Video)**. We greatly appreciate the tremendous effort behind their work!

---

## 📚 Citation

If you find our work or code useful, please cite:

```bibtex
@article{haghighi2026sencache,
  title={SenCache: Accelerating Diffusion Model Inference via Sensitivity-Aware Caching},
  author={Haghighi, Yasaman and Alahi, Alexandre},
  journal={arXiv preprint arXiv:2602.24208},
  year={2026}
}
```
