import os
import sys
import torch

def format_num(num):
    if num >= 1e6:
        return f"{num/1e6:.4f}M"
    elif num >= 1e3:
        return f"{num/1e3:.1f}K"
    return str(num)

def analyze_checkpoint(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file '{checkpoint_path}' not found.")
        sys.exit(1)
        
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    
    # Extract hyperparameters
    n_embd = checkpoint.get('n_embd', 128)
    n_head = checkpoint.get('n_head', 4)
    n_layer = checkpoint.get('n_layer', 3)
    block_size = checkpoint.get('block_size', 128)
    chars = checkpoint.get('chars', [])
    vocab_size = len(chars)
    
    print("==================================================")
    print("           MICRON LLM ARCHITECTURE STATS          ")
    print("==================================================")
    print(f"Vocabulary Size:          {vocab_size} characters")
    print(f"Embedding Dimension (C):  {n_embd}")
    print(f"Attention Heads:          {n_head} (head size: {n_embd // n_head})")
    print(f"Transformer Layers:       {n_layer}")
    print(f"Context Length (T):       {block_size} tokens/chars")
    print(f"Device Compatibility:     CPU & CUDA (RTX 3050 Optimized)")
    print("-" * 50)
    
    # Categorize parameters
    params_breakdown = {
        "Token Embeddings": 0,
        "Position Embeddings": 0,
        "Self-Attention (Q,K,V)": 0,
        "Self-Attention Projections": 0,
        "Feed-Forward Networks": 0,
        "Layer Normalizations": 0,
        "LM Head": 0
    }
    
    total_params = 0
    
    for name, tensor in state_dict.items():
        numel = tensor.numel()
        total_params += numel
        
        if "token_embedding_table" in name:
            params_breakdown["Token Embeddings"] += numel
        elif "position_embedding_table" in name:
            params_breakdown["Position Embeddings"] += numel
        elif "blocks" in name:
            if "sa_head" in name:
                if "key" in name or "query" in name or "value" in name:
                    params_breakdown["Self-Attention (Q,K,V)"] += numel
                elif "proj" in name:
                    params_breakdown["Self-Attention Projections"] += numel
            elif "ffwd" in name:
                params_breakdown["Feed-Forward Networks"] += numel
            elif "ln1" in name or "ln2" in name:
                params_breakdown["Layer Normalizations"] += numel
        elif "ln_f" in name:
            params_breakdown["Layer Normalizations"] += numel
        elif "lm_head" in name:
            params_breakdown["LM Head"] += numel

    print(f"Layer-by-Layer Parameter Breakdown:")
    for key, val in params_breakdown.items():
        percent = (val / total_params) * 100
        print(f"  - {key:<28} : {format_num(val):>8} ({percent:5.2f}%)")
        
    print("-" * 50)
    print(f"Total Model Parameters:          {total_params:,} ({format_num(total_params)})")
    
    # Compute stats
    # Forward pass FLOPs per token approximation: ~ 2 * number of parameters
    # Backward pass FLOPs per token approximation: ~ 4 * number of parameters
    flops_fw_token = 2 * total_params
    flops_bw_token = 4 * total_params
    flops_total_token = flops_fw_token + flops_bw_token
    
    print("\n==================================================")
    print("            COMPUTATIONAL ESTIMATES               ")
    print("==================================================")
    print(f"FLOPs per token (Forward Pass):   {format_num(flops_fw_token)} FLOPs")
    print(f"FLOPs per token (Backward Pass):  {format_num(flops_bw_token)} FLOPs")
    print(f"Total FLOPs per token trained:    {format_num(flops_total_token)} FLOPs")
    
    # VRAM estimation (approximate weights memory + activation memory)
    # Weights in Float32 = 4 bytes per parameter
    weights_mem_mb = (total_params * 4) / (1024 * 1024)
    print(f"Weight File Size on Disk:         {weights_mem_mb:.2f} MB")
    
    # VRAM activation footprint estimation during training (per batch size 16)
    # T = 128, B = 16, C = 128, L = 3
    # MHA + FFN activations:
    # A single forward activation takes approx: B * T * C * L * 10 elements = 16 * 128 * 128 * 3 * 10 * 4 bytes = ~3.1MB VRAM
    # So it is extremely VRAM light!
    activation_mem_mb = (16 * block_size * n_embd * n_layer * 10 * 4) / (1024 * 1024)
    print(f"Estimated VRAM (Weights only):    {weights_mem_mb:.2f} MB")
    print(f"Estimated Activation VRAM (B=16):  {activation_mem_mb:.2f} MB")
    print("==================================================")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.abspath(os.path.join(script_dir, "..", "checkpoints", "micron_model.pt"))
    analyze_checkpoint(checkpoint_path)
