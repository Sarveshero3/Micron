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

### 4. Bits-Per-Character (BPC) as standard metric
* **Insight:** In character-level modeling, word-level perplexity estimation can be misleading due to variable word lengths and tokenization definitions.
* **Lesson:** Bits-per-character (BPC) represents the Shannon entropy of predictions in base 2 and is the correct objective standard. Carving out a clean 5% test set from the held-out validation segment showed a Test BPC of 1.8421 compared to Validation BPC of 1.8267, verifying stable generalization without split leakage.

### 5. Temperature and Top-K Diversity Dynamics
* **Insight:** Autoregressive models behave very differently under different sampling parameters. Greedy decoding collapses into repeating patterns, while high-temperature decoding becomes chaotic.
* **Lesson:** Running a sweep confirmed that at low temperatures (0.2) or Top-K=1, the model collapses into repetition (0.72 repetition rate). High temperatures (1.2) increase perplexity (4.57) representing a loss of structural coherence. Optimal sampling lies in the mid-range (Temp=0.5, Top-K=5) where the repetition rate drops to 0.00 and Self-BLEU-4 is low (0.0242), indicating diverse, fluent outputs.

### 6. Memorization Rates in Small Autoregressive Models
* **Insight:** Small transformers (10.8M parameters) trained on highly structured data can sometimes memorize sequences verbatim.
* **Lesson:** Running a sliding-window Levenshtein search (window length 30, similarity threshold >= 90%) over generated samples against the unique training corpus blocks yielded a 0.00% memorization rate. This indicates that despite its low validation loss, the model acts as a probabilistic generator, synthesizing novel sentence boundaries rather than reproducing training logs verbatim.

### 7. Count-based Baselines and Parameter Scaling Curves
* **Insight:** Evaluating models against simple count-based systems prevents architectural confirmation bias and maps return on model capacity scaling.
* **Lesson:** Baseline evaluations showed BPC scores of 3.31 (Bigram Count) and 2.21 (Trigram Count). A simple neural lookup table bigram model reached 3.45 BPC. Converged Micron (10.8M params, 10k steps) reached 1.27 BPC, proving massive gains from deep contextual self-attention. Training model scaling variants from 1M to 25M parameters for 40 steps mapped early power-law scaling (loss dropping from 2.61 to 2.35), though larger models require more initialization steps to amortize training overhead.

### 8. Noise Fragility in Character-Level Models
* **Insight:** Sub-word BPE tokenizers cushion minor spelling errors. Character-level models evaluate each byte/char directly, magnifying noise.
* **Lesson:** Introducing a small 5% character typo rate (swaps/deletions) causes prediction loss to skyrocket from 1.37 to 6.25, demonstrating extreme fragility. However, context window scaling tests proved that loss monotonically drops from 1.57 (10-char context) to 1.23 (200-char context), validating that the model successfully absorbs long-context history to improve predictions.

### 9. Dynamic Quantization and Quadratic Attention FLOPs
* **Insight:** In small context-length transformers, attention math can be heavily dominated by the quadratic sequence length term rather than Feed-Forward parameters.
* **Lesson:** Analytical FLOP counts showed that multi-head attention (101.8 MFLOPs per block) dominates the FFN (2.36 MFLOPs) by 43x due to the $O(T^2 d)$ attention matrix multiplication. Dynamic INT8 quantization on CPU reduced model size by 3.8x (41.4MB to 10.9MB) with a minimal validation loss delta of 0.0153. However, it resulted in a 0.81x speedup (actually slowing down generation) due to the quantization/dequantization overhead in memory-bandwidth-bound 10.8M parameter models.






