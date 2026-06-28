import os
import sys
import json
import time
import math
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

# Add src folder to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "..", "src")))
from Micron import MicronGPT, device

# -----------------
# 1. Noise Injector Heuristic
# -----------------
def inject_typo_noise(text, rate, chars_list):
    if rate <= 0.0:
        return text
        
    chars = list(text)
    num_noisy = int(len(chars) * rate)
    indices = random.sample(range(len(chars)), min(num_noisy, len(chars)))
    
    for idx in indices:
        action = random.choice(["swap", "delete", "replace"])
        if action == "delete" and len(chars) > 5:
            chars[idx] = ""
        elif action == "swap" and idx < len(chars) - 1:
            # Swap with next character
            chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
        else: # replace
            chars[idx] = random.choice(chars_list)
            
    return "".join(chars)

# -----------------
# 2. Coherence Overlap Heuristic
# -----------------
def compute_coherence_overlap(turns):
    stop_words = {"i", "you", "a", "the", "to", "is", "of", "and", "in", "it", "my", "me", "we", "he", "she", "they", "was", "were", "be", "do", "have", "for", "that"}
    
    def get_keywords(text):
        import re
        cleaned = re.sub(r'[^\w\s]', '', text.lower())
        words = cleaned.split()
        return set([w for w in words if w not in stop_words and len(w) > 2])
        
    keywords_per_turn = [get_keywords(t) for t in turns]
    
    overlaps = []
    for i in range(len(keywords_per_turn) - 1):
        k1 = keywords_per_turn[i]
        k2 = keywords_per_turn[i+1]
        union = k1.union(k2)
        if union:
            overlap_ratio = len(k1.intersection(k2)) / len(union)
        else:
            overlap_ratio = 0.0
        overlaps.append(overlap_ratio)
        
    return sum(overlaps) / len(overlaps) if overlaps else 0.0

def evaluate_robustness():
    print("--------------------------------------------------")
    checkpoint_path = os.path.abspath(os.path.join(script_dir, "..", "checkpoints", "micron_model.pt"))
    dataset_path = os.path.abspath(os.path.join(script_dir, "..", "data", "qa_dataset.txt"))
    output_json_path = os.path.abspath(os.path.join(script_dir, "..", "results", "robustness_results.json"))
    output_plot_path = os.path.abspath(os.path.join(script_dir, "..", "results", "robustness_diagnostic.png"))

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file '{checkpoint_path}' not found.")
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
    
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda L: "".join([itos[i] for i in L])

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

    # Load validation corpus to extract prompts from
    with open(dataset_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    val_data = data[n:]
    
    # -----------------
    # Test 1: Context Limit Sensitivity
    # -----------------
    print("Testing Context Limit Sensitivity...")
    context_lengths = [10, 20, 50, 100, 150, 200, 240]
    context_losses = []
    
    random.seed(42)
    # CPU optimized: evaluate on 30 random sub-segments for each length
    num_samples = 30
    
    for cl in context_lengths:
        losses = []
        for _ in range(num_samples):
            # Extract random start index
            start_idx = random.randint(0, len(val_data) - cl - 2)
            x = val_data[start_idx : start_idx + cl].unsqueeze(0).to(device)
            y = val_data[start_idx + 1 : start_idx + cl + 1].unsqueeze(0).to(device)
            
            with torch.no_grad():
                _, loss = model(x, y)
                losses.append(loss.item())
        mean_loss = sum(losses) / len(losses)
        context_losses.append(mean_loss)
        print(f"  Context Length = {cl:3d} | Mean CE Loss = {mean_loss:.4f}")

    # -----------------
    # Test 2: Noise Injection Sensitivity
    # -----------------
    print("\nTesting Typo Noise Tolerance...")
    noise_rates = [0.0, 0.05, 0.10, 0.20]
    noise_losses = []
    
    for nr in noise_rates:
        losses = []
        for _ in range(num_samples):
            # Extract prompt of size 100
            start_idx = random.randint(0, len(val_data) - 102)
            segment_tokens = val_data[start_idx : start_idx + 100].tolist()
            segment_str = decode(segment_tokens)
            
            # Inject noise into string
            noisy_str = inject_typo_noise(segment_str, nr, chars)
            noisy_tokens = encode(noisy_str)[:100]
            
            # Pad or trim if length changes
            if len(noisy_tokens) < 100:
                noisy_tokens += [stoi[' ']] * (100 - len(noisy_tokens))
                
            x = torch.tensor([noisy_tokens], dtype=torch.long, device=device)
            y = val_data[start_idx + 1 : start_idx + 101].unsqueeze(0).to(device)
            
            with torch.no_grad():
                _, loss = model(x, y)
                losses.append(loss.item())
        mean_loss = sum(losses) / len(losses)
        noise_losses.append(mean_loss)
        print(f"  Noise Rate = {nr*100:2.0f}% | Mean CE Loss = {mean_loss:.4f}")

    # -----------------
    # Test 3: Multi-turn Coherence Overlap
    # -----------------
    print("\nSimulating conversations to test turn coherence...")
    coherence_scores = []
    
    # 3-turn prompts
    base_prompts = [
        ["Who are you?", "Are you sure you are not a machine?", "Do you know who created you?"],
        ["How do I write clean code?", "Does the code need comments?", "Can you write an example?"],
        ["Do you have a favorite movie?", "Tell me about the story.", "Why do people like it?"]
    ]
    
    for conv in base_prompts:
        turns = []
        history = ""
        for prompt in conv:
            turns.append(prompt)
            context = history + f"\nQuestion: {prompt}\nAnswer:"
            context_tokens = torch.tensor([encode(context)], dtype=torch.long, device=device)
            
            with torch.no_grad():
                # Generate a short answer
                output_tokens = model.generate(context_tokens[:, -block_size:], max_new_tokens=80, temperature=0.6, top_k=5)
            decoded = decode(output_tokens[0].tolist())
            response = decoded[len(context):].split("***")[0].strip()
            turns.append(response)
            history = context + " " + response + " ***"
            
        overlap = compute_coherence_overlap(turns)
        coherence_scores.append(overlap)
        print(f"  Conversation: {conv[0][:20]}... | Keyword Coherence = {overlap:.4f}")
        
    avg_coherence = sum(coherence_scores) / len(coherence_scores)
    print(f"  Average Turn Coherence Overlap: {avg_coherence:.4f}")

    # -----------------
    # Save numerical data
    # -----------------
    results = {
        "context_limit": {
            "lengths": context_lengths,
            "losses": context_losses
        },
        "noise_tolerance": {
            "rates": noise_rates,
            "losses": noise_losses
        },
        "multi_turn_coherence": {
            "conversations": [bp[0] for bp in base_prompts],
            "coherence_scores": coherence_scores,
            "average_coherence": avg_coherence
        }
    }
    
    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nRobustness results saved to: {output_json_path}")

    # -----------------
    # Generate Plots
    # -----------------
    print("Generating robustness plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Context Limit
    ax1.plot(context_lengths, context_losses, 'o-', color='#e377c2', linewidth=2)
    ax1.set_xlabel('Context Window Length (tokens)')
    ax1.set_ylabel('Mean Cross-Entropy Loss')
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.set_title('Validation Loss vs Context Window Length')
    
    # Plot 2: Noise Injector
    ax2.plot([r * 100 for r in noise_rates], noise_losses, 's-', color='#17becf', linewidth=2)
    ax2.set_xlabel('Prompt Noise Injection Rate (%)')
    ax2.set_ylabel('Mean Cross-Entropy Loss')
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.set_title('Validation Loss vs Typo Noise Rate')
    
    plt.suptitle("Micron LLM: Input Robustness & Turn Coherence Diagnostics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=200)
    plt.close()
    print(f"Robustness plots saved to: {output_plot_path}")
    print("--------------------------------------------------")

if __name__ == "__main__":
    evaluate_robustness()
