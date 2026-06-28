# Micron — Local Dialogue Transformer

A local, self-contained GPT-style autoregressive language model that trains and runs entirely on consumer hardware. Built on top of Andrej Karpathy's `bigram.py` / NanoGPT lineage, then extended with native PyTorch attention kernels, structured path handling, a character repetition penalty, and a full diagnostic evaluation suite.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Architecture Upgrades](#architecture-upgrades)
3. [Pre-Training Results](#pre-training-convergence-and-loss-profile)
4. [Diagnostic Evaluation Suite](#comprehensive-diagnostic-evaluation-results)
5. [How to Run](#how-to-run)

---

## Project Structure

```text
Micron/
├── src/            Core source code — model architecture & chat loop
├── data/           Data compilers & generated text datasets
├── evaluation/      Diagnostic scripts for evaluation
├── results/         Compiled JSON logs, evaluation reports, and charts
├── checkpoints/     Saved model weights (micron_model.pt)
├── notebooks/       In-depth mathematical notebooks
└── README.md
```

### Mathematics & Deep Learning Notes

The mathematical derivations and dimensional proofs live in interactive notebooks under `notebooks/`, kept separate from this README to keep implementation details front and center.

| Notebook                                   | Covers                                                                                                                |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| [`notes.ipynb`](notebooks/notes.ipynb)     | Standard normal distribution initialization, embedding weight space, the lower-triangular matrix multiplication trick |
| [`gpt-dev.ipynb`](notebooks/gpt-dev.ipynb) | Tensor shapes `(B, T, C)`, Query/Key/Value projections, scaling factors, multi-head attention backpropagation         |

---

## Architecture Upgrades

The core transformer blocks replace manual self-attention loops with PyTorch's native `scaled_dot_product_attention` (SDPA), enabling FlashAttention C++ kernels and a reduced VRAM footprint. Mixed Precision Training (AMP) via `torch.amp` runs FP16 autocasting to take advantage of Tensor Cores during training.

For local usability, a streaming CLI chat interface outputs character-by-character. A repetition penalty discounts recently generated non-space characters in the logits to prevent loops, alongside standard temperature and top-k filtering. Vocabulary, hyperparameters, and weights are bundled into a single self-contained checkpoint for simple loading.

---

## Pre-Training Convergence and Loss Profile

Pre-trained for **10,000 steps** on the Cornell Movie-Dialogs corpus — **20.22 hours** on CPU.

| Metric                | Value  |
| --------------------- | ------ |
| Final training loss   | 0.8282 |
| Final validation loss | 0.8817 |
| Generalization gap    | 0.0535 |

![Loss Convergence Plot](results/loss_convergence.png)

---

## Comprehensive Diagnostic Evaluation Results

A full diagnostic suite verifies language modeling accuracy, generation diversity, memorization rates, scaling laws, input robustness, computational complexity, and attention behavior.

### 1. Language Modeling Metrics

| Split                       | Loss   | BPC    |
| --------------------------- | ------ | ------ |
| Validation                  | 1.2662 | 1.8267 |
| Test                        | 1.2769 | 1.8421 |
| Pre-training val (step 10k) | 0.8817 | 1.2720 |

BPC split leakage delta: **0.0154** — the narrow gap between validation and test BPC confirms stable evaluation partitions.

![Loss and Gradient Smoothness](results/loss_smoothness.png)

### 2. Generation Quality and Diversity

| Metric              | Value  |
| ------------------- | ------ |
| Distinct-1 ratio    | 0.3800 |
| Distinct-2 ratio    | 0.7067 |
| Distinct-3 ratio    | 0.8252 |
| Average Self-BLEU-4 | 0.0242 |

At temperature 0.2 / top-k 1, generation collapses into repetition (rate **0.7273**). Balanced sampling at temperature 0.5 / top-k 5 drops repetition to **0.0000** with a perplexity of **2.25**.

![Diversity Sweep Heatmaps](results/diversity_sweep.png)

### 3. Memorization Check

30 generated 30-character windows were scanned against the training corpus with a Levenshtein sliding window.

- Verbatim / near-verbatim match rate: **0.00%** (0 of 30 samples within edit distance ≤ 3)

This indicates Micron synthesizes novel dialogue from learned syntax rather than copying training data verbatim.

### 4. Count-Based Baselines and Parameter Scaling

| Model                                         | Validation Loss | BPC        |
| --------------------------------------------- | --------------- | ---------- |
| Count bigram baseline                         | 2.2927          | 3.3077     |
| Count trigram baseline                        | 1.5310          | 2.2088     |
| Lookup table bigram                           | 2.3930          | 3.4524     |
| Tiny (~464k params)                           | 2.6144          | 3.7717     |
| Small (~3.3M params)                          | 2.3935          | 3.4530     |
| Medium (~10.8M params)                        | 2.3564          | 3.3995     |
| Large (~25.5M params)                         | 2.3668          | 3.4145     |
| **Fully converged Micron (10.8M, 10k steps)** | **0.8817**      | **1.2720** |

![Scaling Laws and Baselines](results/scaling_laws.png)

### 5. Input Robustness and Turn Coherence

| Test                            | Result                                                       |
| ------------------------------- | ------------------------------------------------------------ |
| Context length: 10 → 200 chars  | Loss 1.5691 → 1.2322 (rises slightly to 1.2456 at 240 chars) |
| Typo noise: 0% → 5% → 10% → 20% | Loss 1.3730 → 6.2538 → 6.9506 → 7.0436                       |
| Multi-turn coherence overlap    | 0.0236 average                                               |

Increasing context consistently improves loss, confirming the model uses contextual history. Character-level attention is highly sensitive to spelling disruption.

![Robustness Diagnostics](results/robustness_diagnostic.png)

### 6. Computational Complexity and Quantization

**Forward FLOPs per token: 625.32 MFLOPs**

| Component             | FLOPs                                                  |
| --------------------- | ------------------------------------------------------ |
| Attention (per block) | 101.84 MFLOPs — dominates by 43x due to O(T²d) scaling |
| FFN (per block)       | 2.36 MFLOPs                                            |
| LM head               | 0.11 MFLOPs                                            |

**Dynamic INT8 quantization**

|                 | FP32        | INT8                         |
| --------------- | ----------- | ---------------------------- |
| Disk size       | 41.40 MB    | 10.90 MB (3.8x smaller)      |
| Validation loss | 1.2543      | 1.2696 (Δ 0.0153)            |
| CPU throughput  | 91.41 tok/s | 73.85 tok/s (0.81x — slower) |

INT8 quantization slows throughput here because a 10M-parameter network at this size is memory-bandwidth bound, not compute bound.

**Training cost:** 1.314 kWh over 20.22 hours (65W CPU TDP) — **$0.21** at $0.16/kWh, **0.499 kg CO2e**.

### 7. Attention Interpretability Maps

Extracted for the prompt `"Question: Do you know my mom?\nAnswer: Yes."`

- **Heads 1, 2, 4** — local diagonal focus, attending to the immediately preceding token
- **Heads 3, 5, 6** — attend to structural tokens (newlines, punctuation), mapping syntax hierarchy

![Attention Maps](results/attention_maps.png)

---

## How to Run

All commands run from the project root.

### 1. Download and compile the dialogue dataset

Downloads the Cornell Movie-Dialogs Corpus and compiles 221,282 QA exchanges.

```cmd
python data/download_corpus.py
```

### 2. Generate the custom Q&A dataset

```cmd
python data/generate_dataset.py
```

### 3. Train or fine-tune the model

Trains on `data/qa_dataset.txt` for 10,000 steps by default, saving to `checkpoints/micron_model.pt`.

```cmd
python src/Micron.py --train
```

Fine-tune an existing checkpoint on the custom dataset (~2 minutes):

```cmd
python src/Micron.py --train --resume --max_iters 1200 --lr 1e-4
```

### 4. Chat with Micron

```cmd
python src/Micron.py --chat
```

### 5. Run the diagnostic evaluation suite

```cmd
python evaluation/eval_bpc.py
python evaluation/eval_diversity.py
python evaluation/eval_memorization.py
python evaluation/eval_baselines.py
python evaluation/eval_robustness.py
python evaluation/eval_efficiency.py
python evaluation/eval_attention.py
```

Legacy baseline benchmarks and plots:

```cmd
python src/benchmark.py
python evaluation/plot_intelligence.py
python evaluation/plot_domain_eval.py
```
