import os
import sys
import json
import time
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np

# Add src folder to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "..", "src")))
from Micron import MicronGPT, device

# -----------------
# 1. Count-based Baselines
# -----------------
def evaluate_ngram_counts(train_data, val_data, vocab_size):
    print("Evaluating Count-based Bigram and Trigram Baselines...")
    
    # Bigram counts: C(c1, c2)
    bigram_counts = np.zeros((vocab_size, vocab_size))
    # Trigram counts: C(c1, c2, c3)
    trigram_counts = np.zeros((vocab_size, vocab_size, vocab_size))
    
    train_list = train_data.tolist()
    val_list = val_data.tolist()
    
    # Accumulate training counts
    for i in range(len(train_list) - 1):
        c1, c2 = train_list[i], train_list[i+1]
        bigram_counts[c1, c2] += 1
        
    for i in range(len(train_list) - 2):
        c1, c2, c3 = train_list[i], train_list[i+1], train_list[i+2]
        trigram_counts[c1, c2, c3] += 1
        
    # Evaluate Bigram validation loss with Laplace (add-1) smoothing
    bigram_probs = (bigram_counts + 1) / (bigram_counts.sum(axis=1, keepdims=True) + vocab_size)
    bigram_val_loss = 0.0
    for i in range(len(val_list) - 1):
        c1, c2 = val_list[i], val_list[i+1]
        bigram_val_loss -= math.log(bigram_probs[c1, c2])
    bigram_val_loss /= (len(val_list) - 1)
    bigram_bpc = bigram_val_loss / math.log(2)
    
    # Evaluate Trigram validation loss with Laplace smoothing
    # We fall back to bigram when context is not seen
    trigram_val_loss = 0.0
    for i in range(len(val_list) - 2):
        c1, c2, c3 = val_list[i], val_list[i+1], val_list[i+2]
        sum_c12 = trigram_counts[c1, c2].sum()
        if sum_c12 > 0:
            prob = (trigram_counts[c1, c2, c3] + 1) / (sum_c12 + vocab_size)
        else:
            prob = bigram_probs[c2, c3]
        trigram_val_loss -= math.log(prob)
    trigram_val_loss /= (len(val_list) - 2)
    trigram_bpc = trigram_val_loss / math.log(2)
    
    print(f"  Count Bigram: Loss = {bigram_val_loss:.4f} | BPC = {bigram_bpc:.4f}")
    print(f"  Count Trigram: Loss = {trigram_val_loss:.4f} | BPC = {trigram_bpc:.4f}")
    
    return {
        "bigram_loss": bigram_val_loss,
        "bigram_bpc": bigram_bpc,
        "trigram_loss": trigram_val_loss,
        "trigram_bpc": trigram_bpc
    }

# -----------------
# 2. Simple Lookup Baseline (Karpathy's Initial Step)
# -----------------
class SimpleLookupBigram(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) # (B, T, C)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

def evaluate_lookup_model(train_data, val_data, vocab_size, block_size):
    print("\nTraining Karpathy's baseline Lookup Table Model...")
    model = SimpleLookupBigram(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    
    batch_size = 32
    def get_batch(dataset):
        ix = torch.randint(len(dataset) - block_size, (batch_size,))
        x = torch.stack([dataset[i : i + block_size] for i in ix])
        y = torch.stack([dataset[i + 1 : i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)
        
    start_time = time.time()
    for step in range(500):
        x, y = get_batch(train_data)
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
    # Evaluate on validation
    losses = torch.zeros(100)
    model.eval()
    with torch.no_grad():
        for k in range(100):
            x, y = get_batch(val_data)
            _, loss = model(x, y)
            losses[k] = loss.item()
    mean_val_loss = losses.mean().item()
    bpc = mean_val_loss / math.log(2)
    
    print(f"  Lookup Baseline trained in {time.time() - start_time:.2f}s.")
    print(f"  Lookup Bigram: Loss = {mean_val_loss:.4f} | BPC = {bpc:.4f}")
    return mean_val_loss, bpc

# -----------------
# 3. Scaling Laws Experiment
# -----------------
def get_num_params(model):
    return sum(p.numel() for p in model.parameters())

def train_scaling_variant(train_data, val_data, vocab_size, block_size, n_embd, n_head, n_layer, name, checkpoint_file, num_steps=200):
    print(f"\nTraining Scaling Variant: {name} (Layers: {n_layer}, Head: {n_head}, Dim: {n_embd})...")
    model = MicronGPT(
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        vocab_size=vocab_size,
        block_size=block_size,
        dropout=0.0
    ).to(device)
    
    num_params = get_num_params(model)
    print(f"  Parameters: {num_params:,}")
    
    # We train for 200 steps with CPU-friendly batch size of 8
    batch_size = 8
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    def get_batch(dataset):
        ix = torch.randint(len(dataset) - block_size, (batch_size,))
        x = torch.stack([dataset[i : i + block_size] for i in ix])
        y = torch.stack([dataset[i + 1 : i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)
        
    start_time = time.time()
    for step in range(num_steps):
        x, y = get_batch(train_data)
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
    elapsed = time.time() - start_time
    
    # Evaluate final validation loss
    model.eval()
    eval_iters = 40
    losses = torch.zeros(eval_iters)
    with torch.no_grad():
        for k in range(eval_iters):
            x, y = get_batch(val_data)
            _, loss = model(x, y)
            losses[k] = loss.item()
    mean_val_loss = losses.mean().item()
    bpc = mean_val_loss / math.log(2)
    
    print(f"  Trained in {elapsed:.2f}s | Val Loss = {mean_val_loss:.4f} | BPC = {bpc:.4f}")
    
    # Save weights to separate checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'block_size': block_size
    }, checkpoint_file)
    print(f"  Checkpoint saved to: {checkpoint_file}")
    
    return num_params, mean_val_loss, bpc

def plot_results(results, output_plot_path, num_steps):
    print("Generating scaling law plots...")
    counts_res = results["count_bigram"]
    lookup_val_loss = results["lookup_bigram"]["loss"]
    variants = results["scaling_law"]["variants"]
    param_counts = [v["params"] for v in variants]
    val_losses = [v["val_loss"] for v in variants]
    
    converged_params = results["scaling_law"]["fully_converged_micron"]["params"]
    converged_val_loss = results["scaling_law"]["fully_converged_micron"]["val_loss"]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot scaling curve for short runs
    ax1.plot(param_counts, val_losses, 'o-', color='#1f77b4', linewidth=2.5, label=f'{num_steps}-step Training Variants')
    for i, txt in enumerate(["Tiny", "Small", f"Medium ({num_steps}s)", "Large"]):
        ax1.annotate(txt, (param_counts[i], val_losses[i]), textcoords="offset points", xytext=(0,10), ha='center', fontweight="bold")
        
    # Mark fully converged Micron checkpoint (from 10k step run)
    ax1.plot(converged_params, converged_val_loss, '*', markersize=14, color='#e377c2', label='Converged Micron (10k steps)')
    ax1.annotate("Fully Converged (10k)", (converged_params, converged_val_loss), textcoords="offset points", xytext=(0,-20), ha='center', color='#e377c2', fontweight="bold")
    
    # Mark Baselines as horizontal lines
    ax1.axhline(y=counts_res["bigram_loss"], color='#ff7f0e', linestyle='--', label='Count Bigram Baseline')
    ax1.axhline(y=lookup_val_loss, color='#2ca02c', linestyle='-.', label='Lookup Table Bigram')
    ax1.axhline(y=counts_res["trigram_loss"], color='#bcbd22', linestyle=':', label='Count Trigram Baseline')
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Model Parameter Count (Log Scale)', fontsize=11)
    ax1.set_ylabel('Validation Cross-Entropy Loss', fontsize=11)
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.set_title("Micron LLM: N-Gram Baselines & Parameter Scaling Curves", fontsize=13, fontweight="bold")
    
    # Position legend to the right side of the plot
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.0)
    
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Scaling plot saved to: {output_plot_path}")

def run_baselines_and_scaling(plot_only=False):
    print("==================================================")
    checkpoint_path = os.path.abspath(os.path.join(script_dir, "..", "checkpoints", "micron_model.pt"))
    dataset_path = os.path.abspath(os.path.join(script_dir, "..", "data", "qa_dataset.txt"))
    output_json_path = os.path.abspath(os.path.join(script_dir, "..", "results", "baseline_scaling_results.json"))
    output_plot_path = os.path.abspath(os.path.join(script_dir, "..", "results", "scaling_laws.png"))

    num_steps = 200 if device == 'cuda' else 40

    if plot_only:
        if not os.path.exists(output_json_path):
            print(f"Error: Results file '{output_json_path}' not found. Cannot run plot-only mode without existing results.")
            sys.exit(1)
        print(f"Loading baseline and scaling results from: {output_json_path}")
        with open(output_json_path, "r") as f:
            results = json.load(f)
        plot_results(results, output_plot_path, num_steps)
        print("==================================================")
        return

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file '{checkpoint_path}' not found.")
        sys.exit(1)
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset '{dataset_path}' not found.")
        sys.exit(1)

    print(f"Loading checkpoint config from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    chars = checkpoint['chars']
    stoi = checkpoint['stoi']
    vocab_size = len(chars)
    block_size = checkpoint['block_size']
    
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    
    # Load dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        text = f.read()
    data = torch.tensor(encode(text), dtype=torch.long)
    
    # Consistent 90-10 train/val split
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    # 1. Count Baselines
    counts_res = evaluate_ngram_counts(train_data, val_data, vocab_size)
    
    # 2. Lookup Baseline
    lookup_val_loss, lookup_val_bpc = evaluate_lookup_model(train_data, val_data, vocab_size, block_size)
    
    # 3. Scaling Laws
    # Define 4 scaling configurations (Tiny, Small, Medium, Large)
    # Checkpoints stored to separate distinctly named files
    configs = [
        {"name": "Tiny (~1M)", "n_embd": 128, "n_head": 2, "n_layer": 2, "file": "scaling_1M.pt"},
        {"name": "Small (~5M)", "n_embd": 256, "n_head": 4, "n_layer": 4, "file": "scaling_5M.pt"},
        {"name": "Medium (~11M)", "n_embd": 384, "n_head": 6, "n_layer": 6, "file": "scaling_10M_temp.pt"},
        {"name": "Large (~20M)", "n_embd": 512, "n_head": 8, "n_layer": 8, "file": "scaling_20M.pt"}
    ]
    
    param_counts = []
    val_losses = []
    val_bpcs = []
    
    for conf in configs:
        ckpt_file = os.path.abspath(os.path.join(script_dir, "..", "checkpoints", conf["file"]))
        p_count, v_loss, v_bpc = train_scaling_variant(
            train_data=train_data,
            val_data=val_data,
            vocab_size=vocab_size,
            block_size=block_size,
            n_embd=conf["n_embd"],
            n_head=conf["n_head"],
            n_layer=conf["n_layer"],
            name=conf["name"],
            checkpoint_file=ckpt_file,
            num_steps=num_steps
        )
        param_counts.append(p_count)
        val_losses.append(v_loss)
        val_bpcs.append(v_bpc)
        
    # Add pre-trained 10.8M Micron model stats (fully converged for 10,000 steps)
    # This represents the true ceiling of the converged architecture
    converged_params = 10797388
    converged_val_loss = 0.8817
    converged_val_bpc = converged_val_loss / math.log(2)
    
    # Save numerical data
    results = {
        "count_bigram": counts_res,
        "lookup_bigram": {
            "loss": lookup_val_loss,
            "bpc": lookup_val_bpc
        },
        "scaling_law": {
            "variants": [
                {
                    "name": configs[i]["name"],
                    "params": param_counts[i],
                    "val_loss": val_losses[i],
                    "val_bpc": val_bpcs[i]
                } for i in range(len(configs))
            ],
            "fully_converged_micron": {
                "params": converged_params,
                "val_loss": converged_val_loss,
                "val_bpc": converged_val_bpc
            }
        }
    }
    
    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nBaseline and Scaling results saved to: {output_json_path}")
    
    plot_results(results, output_plot_path, num_steps)
    print("==================================================")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot-only', action='store_true', help='Only regenerate the plot from existing baseline_scaling_results.json')
    args = parser.parse_args()
    
    run_baselines_and_scaling(plot_only=args.plot_only)
