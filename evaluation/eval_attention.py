import os
import sys
import json
import time
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

# Add src folder to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "..", "src")))
from Micron import MicronGPT, MultiHeadAttention, device

# -----------------
# Python Monkey Patch for Multi-Head Attention
# -----------------
# Since scaled_dot_product_attention runs natively in C++, it does not expose 
# the intermediate T x T attention weight matrix. We override the forward pass 
# to calculate it manually and cache the weights for visualization.
def patched_forward(self, x):
    B, T, C = x.shape
    # Project queries, keys, and values, then reshape to (B, num_heads, T, head_size)
    k = self.key(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
    q = self.query(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
    v = self.value(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
    
    # Manual attention matrix calculation: (B, num_heads, T, head_size) @ (B, num_heads, head_size, T) -> (B, num_heads, T, T)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    
    # Apply causal mask
    tril = torch.tril(torch.ones(T, T, device=x.device))
    att = att.masked_fill(tril[:T, :T] == 0, float('-inf'))
    
    # Compute Softmax probabilities
    att = F.softmax(att, dim=-1)
    
    # Cache attention weights on the module instance (detach to prevent memory leak)
    self.attention_weights = att.detach().cpu()
    
    # Weighted aggregation: (B, num_heads, T, T) @ (B, num_heads, T, head_size) -> (B, num_heads, T, head_size)
    out = att @ v
    
    # Concatenate heads and apply output projection
    out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_size)
    out = self.proj(out)
    return out

# Apply monkey patch to MultiHeadAttention class
MultiHeadAttention.forward = patched_forward

def extract_and_plot_attention():
    print("--------------------------------------------------")
    checkpoint_path = os.path.abspath(os.path.join(script_dir, "..", "checkpoints", "micron_model.pt"))
    output_plot_path = os.path.abspath(os.path.join(script_dir, "..", "results", "attention_maps.png"))

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file '{checkpoint_path}' not found.")
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
    
    encode = lambda s: [stoi[c] for c in s if c in stoi]

    # Recreate model
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

    # Define a clean prompt to visualize attention maps
    prompt_str = "Question: Do you know my mom?\nAnswer: Yes."
    prompt_tokens = encode(prompt_str)
    T = len(prompt_tokens)
    
    x = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    
    print(f"Running forward pass for prompt: '{prompt_str}' ({T} characters)...")
    with torch.no_grad():
        _ = model(x)
        
    # Extract weights from the final block's attention head
    # Shape: (1, num_heads, T, T) -> we extract [0] to get (num_heads, T, T)
    last_block = model.blocks[-1]
    if not hasattr(last_block.sa_head, 'attention_weights'):
        print("Error: Patched attention weights not found. Overwrite failed.")
        sys.exit(1)
        
    attn_matrices = last_block.sa_head.attention_weights[0]  # (n_head, T, T)
    print(f"Extracted attention weights shape: {attn_matrices.shape}")

    # Plot a 2x3 grid of heatmaps for the 6 heads
    print("Generating attention head weight heatmaps...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    char_labels = list(prompt_str)
    
    # We display special characters like space as \u2423 for visualization clarity
    display_labels = [c if c != ' ' else '␣' for c in char_labels]
    
    for h in range(n_head):
        row = h // 3
        col = h % 3
        ax = axes[row, col]
        
        # Plot attention matrix
        im = ax.imshow(attn_matrices[h].numpy(), cmap="viridis", aspect="auto")
        
        # Labeled axes with character text
        ax.set_xticks(range(T))
        ax.set_xticklabels(display_labels, fontsize=8, rotation=90)
        ax.set_yticks(range(T))
        ax.set_yticklabels(display_labels, fontsize=8)
        
        ax.set_title(f"Head {h + 1}")
        
    plt.suptitle("Micron LLM: Self-Attention Weight Maps (Final Layer)", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=200)
    plt.close()
    
    print(f"Attention heatmaps saved successfully to: {output_plot_path}")
    print("--------------------------------------------------")

if __name__ == "__main__":
    extract_and_plot_attention()
