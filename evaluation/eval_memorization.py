import os
import sys
import json
import time
import random
import torch
import sys

# Add src folder to path to reuse tokenizer and model
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "..", "src")))
from Micron import MicronGPT, device

# Standard Levenshtein distance calculation
def edit_distance(s1, s2):
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
        
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
        
    return previous_row[-1]

def evaluate_memorization():
    print("--------------------------------------------------")
    checkpoint_path = os.path.abspath(os.path.join(script_dir, "..", "checkpoints", "micron_model.pt"))
    dataset_path = os.path.abspath(os.path.join(script_dir, "..", "data", "qa_dataset.txt"))
    output_json_path = os.path.abspath(os.path.join(script_dir, "..", "results", "memorization_results.json"))

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

    # Load training text
    print("Loading training corpus...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        training_text = f.read()

    # Extract unique dialogues to prune the edit distance search space
    # The dataset contains repeated QA blocks separated by ***
    raw_dialogues = training_text.split("***")
    unique_dialogues = list(set([d.strip() for d in raw_dialogues if d.strip()]))
    print(f"Loaded {len(raw_dialogues)} dialogue blocks. Found {len(unique_dialogues)} unique dialogues.")

    # Generate 30 sample responses to test
    print("Generating 30 test responses for memorization check...")
    # Seed prompt from test questions
    questions_path = os.path.abspath(os.path.join(script_dir, "..", "data", "test_questions.txt"))
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]
        
    random.seed(1337)
    test_prompts = [random.choice(questions) for _ in range(30)]
    
    generated_samples = []
    for idx, q in enumerate(test_prompts):
        context = f"Question: {q}\nAnswer:"
        context_tokens = torch.tensor([encode(context)], dtype=torch.long, device=device)
        with torch.no_grad():
            output_tokens = model.generate(context_tokens, max_new_tokens=100, temperature=0.6, top_k=5)
        decoded = decode(output_tokens[0].tolist())
        response = decoded[len(context):].split("***")[0].strip()
        generated_samples.append((q, response))
        if (idx+1) % 10 == 0:
            print(f"  Generated {idx+1}/30 responses...")

    # Memorization methodology:
    # - Sliding window length L = 30 characters
    # - Match criteria: edit distance <= 3 characters (>= 90% character similarity)
    # - Boyer-Moore exact matching runs first as a fast filter.
    # - If no exact match is found, Levenshtein is computed on unique training windows.
    # - We prune comparisons where character bag-of-words intersection is low.
    WINDOW_LEN = 30
    EDIT_THRESHOLD = 3
    
    print("\nScanning generated responses for memorized spans...")
    memorized_count = 0
    memorization_matches = []
    
    start_scan_time = time.time()
    
    for idx, (prompt, response) in enumerate(generated_samples):
        if len(response) < WINDOW_LEN:
            continue
            
        is_memorized = False
        best_match_corpus = ""
        best_match_gen = ""
        best_dist = 999
        
        # Slide a window over the generated response
        for i in range(len(response) - WINDOW_LEN + 1):
            window = response[i:i+WINDOW_LEN]
            
            # Step 1: Boyer-Moore Exact Match Fast Filter
            if window in training_text:
                is_memorized = True
                best_dist = 0
                best_match_gen = window
                # Find matching context in unique dialogues for display
                for diag in unique_dialogues:
                    if window in diag:
                        best_match_corpus = diag
                        break
                break
                
            # Step 2: Pruned Levenshtein search over unique dialogues
            window_chars = set(window)
            for diag in unique_dialogues:
                if len(diag) < WINDOW_LEN:
                    continue
                # If character intersection is too small, edit distance must exceed threshold
                diag_chars = set(diag)
                overlap = len(window_chars.intersection(diag_chars))
                if overlap < 20: # Pruning threshold
                    continue
                    
                # Compute sliding edit distance
                for j in range(len(diag) - WINDOW_LEN + 1):
                    diag_window = diag[j:j+WINDOW_LEN]
                    dist = edit_distance(window, diag_window)
                    if dist < best_dist:
                        best_dist = dist
                        best_match_gen = window
                        best_match_corpus = diag_window
                        
                    if dist <= EDIT_THRESHOLD:
                        is_memorized = True
                        break
                if is_memorized:
                    break
            if is_memorized:
                break
                
        if is_memorized:
            memorized_count += 1
            # Truncate context for cleaner logging
            matching_segment = best_match_corpus[:120] + ("..." if len(best_match_corpus) > 120 else "")
            print(f"  [{idx+1}] MEMORIZED (Edit Distance: {best_dist})")
            print(f"      Prompt:    {prompt}")
            print(f"      Generated: ... {best_match_gen} ...")
            print(f"      Corpus:    ... {matching_segment} ...")
            memorization_matches.append({
                "prompt": prompt,
                "response": response,
                "matched_gen": best_match_gen,
                "matched_corpus": matching_segment,
                "edit_distance": best_dist
            })
        else:
            print(f"  [{idx+1}] Unique generation (No training matches)")
            print(f"      Prompt:    {prompt}")
            print(f"      Generated: {response}")

    scan_time = time.time() - start_scan_time
    memorization_rate = (memorized_count / len(generated_samples)) * 100
    print(f"\nScan completed in {scan_time:.2f}s.")
    print(f"Memorization Rate: {memorization_rate:.2f}% ({memorized_count}/{len(generated_samples)} samples contain memorized spans)")
    
    # Save numerical data
    results = {
        "memorization_rate": memorization_rate,
        "memorized_count": memorized_count,
        "total_samples": len(generated_samples),
        "evaluated_samples": len([s for s in generated_samples if len(s[1]) >= WINDOW_LEN]),
        "window_length": WINDOW_LEN,
        "edit_threshold": EDIT_THRESHOLD,
        "scan_time_seconds": scan_time,
        "all_generations": [{"prompt": q, "response": r, "length": len(r)} for q, r in generated_samples],
        "matches": memorization_matches
    }
    
    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to: {output_json_path}")
    print("--------------------------------------------------")

if __name__ == "__main__":
    evaluate_memorization()
