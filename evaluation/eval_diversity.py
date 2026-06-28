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
import numpy as np

# Add src folder to path to reuse structures
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "..", "src")))
from Micron import MicronGPT, device

# Clean and tokenize text into words for n-gram evaluations
def get_words(text):
    import re
    # Lowercase and split on spaces/punctuation
    cleaned = re.sub(r'[^\w\s]', '', text.lower())
    return cleaned.split()

# Helper to compute n-grams from a list of words
def get_ngrams(words, n):
    ngrams = []
    for i in range(len(words) - n + 1):
        ngrams.append(tuple(words[i:i+n]))
    return ngrams

# Custom BLEU-4 score implementation to avoid external dependencies
def compute_sentence_bleu(candidate, references):
    cand_words = get_words(candidate)
    ref_word_lists = [get_words(ref) for ref in references]
    
    if not cand_words:
        return 0.0
        
    precisions = []
    for n in range(1, 5):
        cand_ngrams = get_ngrams(cand_words, n)
        if not cand_ngrams:
            precisions.append(0.0)
            continue
            
        ref_ngram_counts = []
        for ref_words in ref_word_lists:
            ref_ngrams = get_ngrams(ref_words, n)
            counts = {}
            for ng in ref_ngrams:
                counts[ng] = counts.get(ng, 0) + 1
            ref_ngram_counts.append(counts)
            
        # Count matched n-grams
        cand_counts = {}
        for ng in cand_ngrams:
            cand_counts[ng] = cand_counts.get(ng, 0) + 1
            
        matched = 0
        for ng, count in cand_counts.items():
            max_ref = 0
            for ref_counts in ref_ngram_counts:
                max_ref = max(max_ref, ref_counts.get(ng, 0))
            matched += min(count, max_ref)
            
        precisions.append(matched / len(cand_ngrams))
        
    # Geometric mean with smoothing
    log_sum = 0.0
    for p in precisions:
        if p == 0:
            log_sum += math.log(1e-9)
        else:
            log_sum += math.log(p)
            
    # Weights for BLEU-4 are uniform 0.25
    bleu = math.exp(0.25 * log_sum)
    return bleu

def evaluate_diversity():
    print("--------------------------------------------------")
    checkpoint_path = os.path.abspath(os.path.join(script_dir, "..", "checkpoints", "micron_model.pt"))
    questions_path = os.path.abspath(os.path.join(script_dir, "..", "data", "test_questions.txt"))
    output_json_path = os.path.abspath(os.path.join(script_dir, "..", "results", "diversity_results.json"))
    output_plot_path = os.path.abspath(os.path.join(script_dir, "..", "results", "diversity_sweep.png"))

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file '{checkpoint_path}' not found.")
        sys.exit(1)
    if not os.path.exists(questions_path):
        print(f"Error: Test questions '{questions_path}' not found.")
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

    # Load test questions to serve as prompts
    with open(questions_path, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
        
    # We want 25 generations. We repeat prompts if we have fewer.
    random.seed(42)
    selected_prompts = []
    for _ in range(25):
        selected_prompts.append(random.choice(prompts))

    # 1. Generate text samples for Distinct N-gram and Self-BLEU
    print("Generating 25 samples for diversity evaluations...")
    generated_texts = []
    start_time = time.time()
    
    for idx, prompt in enumerate(selected_prompts):
        context = f"Question: {prompt}\nAnswer:"
        context_tokens = torch.tensor([encode(context)], dtype=torch.long, device=device)
        
        # CPU optimization: limit generated characters to 100 per prompt
        with torch.no_grad():
            output_tokens = model.generate(context_tokens, max_new_tokens=100, temperature=0.6, top_k=5)
        decoded = decode(output_tokens[0].tolist())
        # Extract only the response (everything after the Answer: prompt prefix)
        response = decoded[len(context):].split("***")[0].strip()
        generated_texts.append(response)
        
        if (idx + 1) % 5 == 0:
            print(f"  Generated {idx + 1}/25 samples...")
            
    print(f"Generation of 25 samples completed in {time.time() - start_time:.2f}s.")

    # Compute Distinct-1, Distinct-2, and Distinct-3 gram ratios
    print("Calculating distinct n-gram ratios...")
    all_words = []
    unique_grams = [set(), set(), set()]
    total_grams = [0, 0, 0]
    
    for text in generated_texts:
        words = get_words(text)
        all_words.extend(words)
        for n in range(1, 4):
            ngrams = get_ngrams(words, n)
            total_grams[n-1] += len(ngrams)
            for ng in ngrams:
                unique_grams[n-1].add(ng)
                
    distinct_1 = len(unique_grams[0]) / total_grams[0] if total_grams[0] > 0 else 0.0
    distinct_2 = len(unique_grams[1]) / total_grams[1] if total_grams[1] > 0 else 0.0
    distinct_3 = len(unique_grams[2]) / total_grams[2] if total_grams[2] > 0 else 0.0
    
    print(f"  Distinct-1 ratio: {distinct_1:.4f}")
    print(f"  Distinct-2 ratio: {distinct_2:.4f}")
    print(f"  Distinct-3 ratio: {distinct_3:.4f}")

    # Compute Self-BLEU (BLEU-4 score of each sentence against all other 24 reference sentences)
    print("Calculating Self-BLEU score...")
    bleu_scores = []
    for i in range(len(generated_texts)):
        candidate = generated_texts[i]
        references = [generated_texts[j] for j in range(len(generated_texts)) if j != i]
        score = compute_sentence_bleu(candidate, references)
        bleu_scores.append(score)
        
    avg_self_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    print(f"  Average Self-BLEU-4: {avg_self_bleu:.4f} (lower represents more diverse outputs)")

    # 2. Temperature and Top-K Grid Sweep (Repetition rate & Perplexity)
    print("\nRunning grid sweep over Temperature vs Top-K...")
    # Select a standard test prompt
    test_prompt = "Question: What is your name?\nAnswer:"
    context_tokens = torch.tensor([encode(test_prompt)], dtype=torch.long, device=device)
    
    temps = [0.2, 0.5, 0.8, 1.2]
    top_ks = [1, 5, 20, 100]
    
    rep_heatmap = np.zeros((len(temps), len(top_ks)))
    ppl_heatmap = np.zeros((len(temps), len(top_ks)))
    
    for t_idx, temp in enumerate(temps):
        for k_idx, top_k in enumerate(top_ks):
            # Generate a 60 character response to keep CPU time under 30 seconds
            with torch.no_grad():
                output_tokens = model.generate(context_tokens, max_new_tokens=60, temperature=temp, top_k=top_k)
            decoded = decode(output_tokens[0].tolist())
            response = decoded[len(test_prompt):].split("***")[0].strip()
            
            # Compute Repetition Rate (ratio of repeated 4-character n-grams)
            # Since spelling can loop at char level, char-level 4-grams is highly sensitive
            char_4grams = [response[i:i+4] for i in range(len(response) - 3)]
            unique_char_4grams = set(char_4grams)
            rep_rate = 1.0 - (len(unique_char_4grams) / len(char_4grams)) if char_4grams else 0.0
            rep_heatmap[t_idx, k_idx] = rep_rate
            
            # Compute Perplexity on generated response using the model itself
            # We treat generated response as targets
            response_tokens = torch.tensor([encode(response)], dtype=torch.long, device=device)
            if response_tokens.size(1) > 2:
                with torch.no_grad():
                    logits, loss = model(response_tokens[:, :-1], response_tokens[:, 1:])
                    ppl = math.exp(loss.item()) if loss is not None else 1.0
            else:
                ppl = 1.0
            # Clip high perplexities for visualization scaling
            ppl_heatmap[t_idx, k_idx] = min(ppl, 100.0)
            
            print(f"  Sweep: Temp = {temp:.1f}, Top-K = {top_k:3d} | Rep Rate = {rep_rate:.4f} | Perplexity = {ppl:.2f}")

    # 3. Save numerical results
    results = {
        "distinct_1": distinct_1,
        "distinct_2": distinct_2,
        "distinct_3": distinct_3,
        "avg_self_bleu": avg_self_bleu,
        "temps": temps,
        "top_ks": top_ks,
        "repetition_grid": rep_heatmap.tolist(),
        "perplexity_grid": ppl_heatmap.tolist(),
        "sample_generations": generated_texts[:5] # log first 5 samples
    }
    
    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results written to: {output_json_path}")

    # 4. Generate plots
    print("Generating sweep heatmap plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Repetition Rate Heatmap
    im1 = ax1.imshow(rep_heatmap, cmap="YlOrRd", aspect="auto")
    ax1.set_xticks(range(len(top_ks)))
    ax1.set_xticklabels(top_ks)
    ax1.set_yticks(range(len(temps)))
    ax1.set_yticklabels(temps)
    ax1.set_xlabel("Top-K Value")
    ax1.set_ylabel("Temperature")
    ax1.set_title("N-gram Repetition Rate (Lower is Better)")
    fig.colorbar(im1, ax=ax1, label="Repetition Ratio")
    
    # Loop to add text labels in cells
    for i in range(len(temps)):
        for j in range(len(top_ks)):
            ax1.text(j, i, f"{rep_heatmap[i, j]:.2f}", ha="center", va="center", color="black" if rep_heatmap[i, j] < 0.6 else "white", fontweight="bold")
            
    # Perplexity Heatmap
    im2 = ax2.imshow(ppl_heatmap, cmap="coolwarm", aspect="auto")
    ax2.set_xticks(range(len(top_ks)))
    ax2.set_xticklabels(top_ks)
    ax2.set_yticks(range(len(temps)))
    ax2.set_yticklabels(temps)
    ax2.set_xlabel("Top-K Value")
    ax2.set_ylabel("Temperature")
    ax2.set_title("Model Perplexity on Output (Lower is More Fluent)")
    fig.colorbar(im2, ax=ax2, label="Perplexity (capped at 100)")
    
    for i in range(len(temps)):
        for j in range(len(top_ks)):
            ax2.text(j, i, f"{ppl_heatmap[i, j]:.1f}", ha="center", va="center", color="black" if ppl_heatmap[i, j] < 60.0 else "white", fontweight="bold")
            
    plt.suptitle("Micron LLM: Diversity & Sampling Generation Sweeps", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=200)
    plt.close()
    print(f"Heatmap plots saved to: {output_plot_path}")
    print("--------------------------------------------------")

if __name__ == "__main__":
    evaluate_diversity()
