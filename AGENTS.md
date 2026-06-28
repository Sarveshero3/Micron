# Micron Project Agent Registry

This document records the contributions of the agentic engineering systems that have developed and benchmarked this repository.

---

## Agent Contributions

### Agent 1
* **Role:** Initial Upgrader and Infrastructure Developer
* **Contributions:**
  - Duplicated the baseline `bigram.py` into `Micron.py` (later moved to `src/`).
  - Integrated PyTorch native Scaled Dot-Product Attention (SDPA / FlashAttention).
  - Added mixed-precision training (AMP) using FP16 autocasting and grad scaling.
  - Implemented initial checkpoint serialization and a real-time character-streaming `--chat` loop.
  - Wrote the Cornell Movie-Dialogs corpus dialogue compiler (`download_corpus.py`).

### Agent 2 (Current)
* **Role:** Systems Architect and Evaluation Lead
* **Contributions:**
  - Reorganized project layout into `src/`, `data/`, `evaluation/`, `results/`, `checkpoints/`, and `notebooks/`.
  - Added the training `--resume` flag to allow seamless continuation of weights and AdamW optimizer states from checkpoint.
  - Resolved chat repetition cycles by implementing a character-level repetition penalty in logits sampling.
  - Designed and executed the comprehensive evaluation suite (Core LM Metrics, Diversity, Memorization, Baselines, Robustness, Computational Efficiency, Attention maps).
  - Implemented `evaluation/eval_bpc.py` to evaluate validation and test BPC and gradient norm smoothness.
  - Implemented `evaluation/eval_diversity.py` to calculate distinct n-gram ratios, Self-BLEU, and run temperature/top-k sweeps.
  - Implemented `evaluation/eval_memorization.py` to run Levenshtein sliding window memorization checks on training data.
  - Cleaned and rewrote project documentation (`README.md`, `results/micron_evaluation_report.md`).
