# Micron LLM Engineering Lessons Learned

This living document tracks architectural insights, optimization breakthroughs, and gotchas identified while building and testing this local character-level language model.

---

## Historical Lessons

### 1. FlashAttention Kernel Speedups (SDPA)
* **Insight:** Writing manual multi-head self-attention loops in PyTorch causes high memory fragmentation and redundant kernel launches on the GPU.
* **Lesson:** Replacing custom attention matrix multiplications with PyTorch's native `F.scaled_dot_product_attention` allows the compiler to bind directly to FlashAttention or memory-efficient attention C++ kernels. This reduced GPU VRAM usage and sped up training loops by over 1.5x on the RTX 3050.

### 2. Mixed Precision Stability
* **Insight:** Training in pure FP16 can trigger underflows and NaN gradients in small models.
* **Lesson:** Combining `torch.amp.autocast('cuda')` with `torch.amp.GradScaler('cuda')` is critical. The scaler dynamically scales loss values to prevent underflow, ensuring training stability while utilizing Tensor Cores.

### 3. Logits Repetition Loops in Character Models
* **Insight:** Autoregressive character-level models easily lock into repetitive n-gram loops (e.g. repeating "I don't know") due to self-attending to highly frequent training phrases.
* **Lesson:** Lowering temperatures makes generation too deterministic, exacerbating loops. Introducing a gentle repetition penalty directly in the logits sampling step (reducing logits for recently generated alphanumeric characters) breaks these cycles effectively.
