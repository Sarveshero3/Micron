import os
import sys
import json
import time
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

# Add src folder to path to reuse tokenizer and model structures
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "..", "src")))
from Micron import MicronGPT, device

def evaluate_bpc():
    print("--------------------------------------------------")
    # 1. Resolve paths
    checkpoint_path = os.path.abspath(os.path.join(script_dir, "..", "checkpoints", "micron_model.pt"))
    dataset_path = os.path.abspath(os.path.join(script_dir, "..", "data", "qa_dataset.txt"))
    output_json_path = os.path.abspath(os.path.join(script_dir, "..", "results", "bpc_results.json"))
    output_plot_path = os.path.abspath(os.path.join(script_dir, "..", "results", "loss_smoothness.png"))
    temp_checkpoint_path = os.path.abspath(os.path.join(script_dir, "..", "checkpoints", "micron_temp_gradient.pt"))

    # 2. Check dependencies
    if not os.path.exists(checkpoint_path):
        print(f"Error: Main model checkpoint '{checkpoint_path}' not found.")
        sys.exit(1)
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset '{dataset_path}' not found.")
        sys.exit(1)

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    chars = checkpoint['chars']
    stoi = checkpoint['stoi']
    itos = checkpoint['itos']
    vocab_size = len(chars)
    block_size = checkpoint['block_size']
    n_embd = checkpoint['n_embd']
    n_head = checkpoint['n_head']
    n_layer = checkpoint['n_layer']
    
    # 3. Load dataset
    print(f"Loading dataset from: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    data = torch.tensor(encode(text), dtype=torch.long)
    
    # Verify split logic
    # Training script used n = int(0.9 * len(data)) as train/val cutoff.
    # To check validation leakage, we split the 10% validation split into:
    # - Val split (first 5% of validation split, which is data[90% : 95%])
    # - Test split (remaining 5% of validation split, which is data[95% : 100%])
    # Both splits have never been updated via backpropagation.
    total_len = len(data)
    train_end = int(0.90 * total_len)
    val_end = int(0.95 * total_len)
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    print(f"Data size: {total_len} tokens")
    print(f"  - Train data size: {len(train_data)} tokens (90.0%)")
    print(f"  - Val data size:   {len(val_data)} tokens (5.0%)")
    print(f"  - Test data size:  {len(test_data)} tokens (5.0%)")
    
    # 4. Instantiate and load model
    model = MicronGPT(
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        vocab_size=vocab_size,
        block_size=block_size,
        dropout=0.0
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Helper to calculate loss over batches (optimized for CPU speed)
    batch_size = 64 if device == 'cuda' else 8
    eval_iters = 500 if device == 'cuda' else 50
    train_batch_size = 64 if device == 'cuda' else 8
    
    def get_batch(dataset, b_size):
        ix = torch.randint(len(dataset) - block_size, (b_size,))
        x = torch.stack([dataset[i : i + block_size] for i in ix])
        y = torch.stack([dataset[i + 1 : i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)
        
    @torch.no_grad()
    def calculate_loss_and_bpc(dataset):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(dataset, batch_size)
            _, loss = model(x, y)
            losses[k] = loss.item()
        mean_loss = losses.mean().item()
        bpc = mean_loss / math.log(2)  # BPC = Cross Entropy Loss / ln(2)
        return mean_loss, bpc

    print("Computing metrics on validation and test sets...")
    val_loss, val_bpc = calculate_loss_and_bpc(val_data)
    test_loss, test_bpc = calculate_loss_and_bpc(test_data)
    
    print(f"Validation Loss: {val_loss:.4f} | Validation BPC: {val_bpc:.4f}")
    print(f"Test Loss:       {test_loss:.4f} | Test BPC:       {test_bpc:.4f}")
    
    # 5. Run a 100-step training continuation to log loss & gradient norms
    # Make sure we use a separate temp checkpoint filename to protect the main checkpoint
    print("\nRunning a short 100-step training run to log gradient norm smoothness...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    if 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception:
            # If optimizer state shapes mismatch, initialize fresh
            pass
            
    step_losses = []
    grad_norms = []
    
    for step in range(100):
        x, y = get_batch(train_data, train_batch_size)
        logits, loss = model(x, y)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Calculate l2 gradient norm
        # We sum squared gradients, then take sqrt
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        optimizer.step()
        
        step_losses.append(loss.item())
        grad_norms.append(total_norm)
        
        if step % 20 == 0:
            print(f"  Step {step:3d}: Train Loss = {loss.item():.4f} | Grad Norm = {total_norm:.4f}")
            
    # Save the temporary run checkpoints to a separate file (Never overwrite micron_model.pt!)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iter': checkpoint['iter'] + 100,
        'val_loss': val_loss
    }, temp_checkpoint_path)
    print(f"Temporary gradient checkpoint saved to: {temp_checkpoint_path}")

    # 6. Save results to JSON
    # Parse history steps from user's logs
    history_steps = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000]
    history_val_losses = [5.0958, 1.6469, 1.3556, 1.1157, 1.0509, 1.0070, 0.9836, 0.9676, 0.9519, 0.9418, 0.9306, 0.9222, 0.9184, 0.9173, 0.9033, 0.9020, 0.8988, 0.8918, 0.8938, 0.8817, 0.8817]
    history_val_bpc = [l / math.log(2) for l in history_val_losses]
    
    results = {
        "val_loss": val_loss,
        "val_bpc": val_bpc,
        "test_loss": test_loss,
        "test_bpc": test_bpc,
        "step_losses": step_losses,
        "grad_norms": grad_norms,
        "history_steps": history_steps,
        "history_val_losses": history_val_losses,
        "history_val_bpc": history_val_bpc
    }
    
    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results written to: {output_json_path}")
    
    # 7. Generate plots
    print("Generating loss smoothness and convergence plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Historical Val Loss and BPC
    color = '#1f77b4'
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Validation Loss', color=color)
    ax1.plot(history_steps, history_val_losses, color=color, marker='o', label='Val Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    ax1_twin = ax1.twinx()
    color = '#d62728'
    ax1_twin.set_ylabel('Val Bits-Per-Character (BPC)', color=color)
    ax1_twin.plot(history_steps, history_val_bpc, color=color, linestyle='--', label='Val BPC')
    ax1_twin.tick_params(axis='y', labelcolor=color)
    ax1.set_title("Pre-training Convergence History")
    
    # Plot 2: 100-step gradient norm and loss smoothness
    color = '#2ca02c'
    ax2.set_xlabel('Step (Continuation)')
    ax2.set_ylabel('Step Train Loss', color=color)
    ax2.plot(range(100), step_losses, color=color, label='Loss')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    ax2_twin = ax2.twinx()
    color = '#9467bd'
    ax2_twin.set_ylabel('Gradient L2 Norm', color=color)
    ax2_twin.plot(range(100), grad_norms, color=color, label='Grad Norm', alpha=0.8)
    ax2_twin.tick_params(axis='y', labelcolor=color)
    ax2.set_title("Gradient Norm & Loss Smoothness (100 steps)")
    
    plt.suptitle("Micron LM Core Performance & Optimization Metrics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=200)
    plt.close()
    print(f"Smoothness plots saved to: {output_plot_path}")
    print("--------------------------------------------------")

if __name__ == "__main__":
    evaluate_bpc()
