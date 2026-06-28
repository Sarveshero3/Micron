import os
import sys
import torch
import torch.nn as nn
from torch.nn import functional as F

# Add src folder to path to import Micron
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "..", "src")))
from Micron import MicronGPT

device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_questions(checkpoint_path, questions_path):
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint '{checkpoint_path}' not found. Please train the model first.")
        sys.exit(1)
    if not os.path.exists(questions_path):
        print(f"Error: Questions file '{questions_path}' not found.")
        sys.exit(1)
        
    print(f"Loading model checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    chars = checkpoint['chars']
    stoi = checkpoint['stoi']
    itos = checkpoint['itos']
    vocab_size = len(chars)
    block_size = checkpoint['block_size']
    
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda L: "".join([itos[i] for i in L])
    
    # Recreate the model structure
    model = MicronGPT(
        n_embd=checkpoint['n_embd'],
        n_head=checkpoint['n_head'],
        n_layer=checkpoint['n_layer'],
        vocab_size=vocab_size,
        block_size=block_size,
        dropout=0.0
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"\nEvaluating questions from '{questions_path}' using local model...")
    print("-" * 60)
    
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]
        
    for idx, q in enumerate(questions):
        print(f"[{idx+1}] Question: {q}")
        prompt = f"Question: {q}\nAnswer:"
        context_tokens = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
        
        # Generation loop
        input_seq = context_tokens
        response_text = ""
        
        for _ in range(300): # limit length
            idx_cond = input_seq[:, -block_size:]
            with torch.no_grad():
                if device == 'cuda':
                    with torch.amp.autocast('cuda'):
                        logits, _ = model(idx_cond)
                else:
                    logits, _ = model(idx_cond)
                    
            # Use temperature 0.6 and top_k 2 for stable testing
            logits = logits[:, -1, :] / 0.6
            v, _ = torch.topk(logits, min(2, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            input_seq = torch.cat((input_seq, idx_next), dim=1)
            char_new = decode([idx_next.item()])
            
            response_text += char_new
            if response_text.endswith("***") or response_text.endswith("\n\nQuestion:"):
                if response_text.endswith("***"):
                    response_text = response_text[:-3]
                elif response_text.endswith("\n\nQuestion:"):
                    response_text = response_text[:-11]
                break
                
        print(f"Answer: {response_text.strip()}")
        print("-" * 60)

if __name__ == "__main__":
    checkpoint_path = os.path.abspath(os.path.join(script_dir, "..", "checkpoints", "micron_model.pt"))
    questions_path = os.path.abspath(os.path.join(script_dir, "..", "data", "test_questions.txt"))
    evaluate_questions(checkpoint_path, questions_path)
