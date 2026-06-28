import argparse
import sys
import os
import time
import torch
import torch.nn as nn
from torch.nn import functional as F

# Default hyperparameters
batch_size = 64    # how many independent sequences will we process in parallel?
block_size = 256   # what is the max context length for predictions?
max_iters = 10000  # train for a long time to achieve deep convergence/memorization
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embd = 384       # number of embedding dimensions (large model)
n_head = 6         # 384/6 = 64-dimensional head size (standard)
n_layer = 6        # number of transformer blocks (large model)
dropout = 0.2

# -----------------

# Set seed for reproducibility
torch.manual_seed(1337)

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

class MultiHeadAttention(nn.Module):
    """Vectorized Multi-Head Self-Attention using PyTorch Scaled Dot-Product Attention (FlashAttention)"""

    def __init__(self, num_heads, head_size, block_size, n_embd, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.key = nn.Linear(n_embd, num_heads * head_size, bias=False)
        self.query = nn.Linear(n_embd, num_heads * head_size, bias=False)
        self.value = nn.Linear(n_embd, num_heads * head_size, bias=False)
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout_p = dropout

    def forward(self, x):
        B, T, C = x.shape
        # Project queries, keys, and values, then reshape to (B, num_heads, T, head_size)
        k = self.key(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        q = self.query(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        
        # Native PyTorch FlashAttention (scaled dot product attention with causal masking)
        out = F.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=self.dropout_p if self.training else 0.0, 
            is_causal=True
        ) # (B, num_heads, T, head_size)
        
        # Concatenate heads and apply output projection
        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_size)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    """Feed-Forward Network with modern GELU activation"""

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),  # upgraded from ReLU for better gradient flow
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer Block: communication (attention) followed by computation (feed-forward)"""

    def __init__(self, n_embd, num_heads, block_size, dropout):
        super().__init__()
        self.sa_head = MultiHeadAttention(num_heads, n_embd // num_heads, block_size, n_embd, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Pre-LN architecture (Residual Connections)
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MicronGPT(nn.Module):
    """The main Micron Autoregressive Language Model"""

    def __init__(self, n_embd, n_head, n_layer, vocab_size, block_size, dropout=0.2):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Token and position embeddings
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        
        # Transformer blocks
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # Generate new tokens given a context sequence
        for _ in range(max_new_tokens):
            # crop context to the max block_size
            idx_cond = idx[:, -self.block_size:]
            
            # get logits
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] / temperature
            
            # top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
                
            # sample next token
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# -----------------
# Training Logic
# -----------------

def train_model(args):
    global block_size, batch_size, n_embd, n_head, n_layer, dropout
    
    if not os.path.exists(args.data):
        print(f"Error: Training dataset file '{args.data}' not found.")
        print("Please run `python generate_dataset.py` first to generate the dataset.")
        sys.exit(1)
        
    print(f"Reading training text from {args.data}...")
    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()
        
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"Vocabulary size: {vocab_size} unique characters.")
    
    # Vocabulary mappings
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    
    # Split dataset
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    def get_batch(split):
        dataset = train_data if split == "train" else val_data
        ix = torch.randint(len(dataset) - block_size, (batch_size,))
        x = torch.stack([dataset[i : i + block_size] for i in ix])
        y = torch.stack([dataset[i + 1 : i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate_loss(model):
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                if device == 'cuda':
                    with torch.amp.autocast('cuda'):
                        _, loss = model(X, Y)
                else:
                    _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # Initialize model
    model = MicronGPT(
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        vocab_size=vocab_size,
        block_size=block_size,
        dropout=dropout
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Mixed precision gradient scaler (for RTX 3050 speedups)
    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None
    
    iters = int(args.max_iters) if args.max_iters is not None else max_iters
    best_val_loss = float('inf')
    start_iter = 0
    
    # Resume from checkpoint if requested and file exists
    if args.resume:
        if os.path.exists(args.checkpoint):
            print(f"Resuming training from checkpoint '{args.checkpoint}'...")
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_iter = checkpoint['iter'] + 1
            best_val_loss = checkpoint['val_loss']
            print(f"Successfully loaded checkpoint. Resuming from step {start_iter} (Best Val Loss: {best_val_loss:.4f})")
        else:
            print(f"Warning: Checkpoint file '{args.checkpoint}' not found. Starting from scratch.")
            
    print("\n--------------------------------------------------")
    print(f"Training Micron GPT Model...")
    print(f"Device: {device.upper()}")
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"Sequence Context Length (Block Size): {block_size}")
    print(f"Batch Size: {batch_size}")
    print("--------------------------------------------------\n")
    
    start_time = time.time()
    
    for iter in range(start_iter, iters):
        # Every once in a while evaluate loss
        if iter % eval_interval == 0 or iter == iters - 1:
            losses = estimate_loss(model)
            val_loss = losses['val']
            print(f"Step {iter:4d}: train loss {losses['train']:.4f}, val loss {val_loss:.4f} | Time: {time.time() - start_time:.1f}s")
            
            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'chars': chars,
                    'stoi': stoi,
                    'itos': itos,
                    'iter': iter,
                    'val_loss': val_loss,
                    'n_embd': n_embd,
                    'n_head': n_head,
                    'n_layer': n_layer,
                    'block_size': block_size,
                }
                torch.save(checkpoint, args.checkpoint)
                print(f"  ==> Saved best model checkpoint to '{args.checkpoint}'")
                
        # Sample training batch
        xb, yb = get_batch("train")
        
        # Training step with AMP
        if device == 'cuda':
            with torch.amp.autocast('cuda'):
                logits, loss = model(xb, yb)
        else:
            logits, loss = model(xb, yb)
            
        optimizer.zero_grad(set_to_none=True)
        
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
    print(f"\nTraining completed in {time.time() - start_time:.1f}s.")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved at: {args.checkpoint}")

# -----------------
# Chat Mode Logic
# -----------------

def chat_mode(args):
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint file '{checkpoint_path}' not found.")
        print("Please train the model first by running: python Micron.py --train")
        sys.exit(1)
        
    print(f"Loading Micron local model from '{checkpoint_path}'...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load vocabulary from checkpoint
    chars = checkpoint['chars']
    stoi = checkpoint['stoi']
    itos = checkpoint['itos']
    vocab_size = len(chars)
    block_size_chk = checkpoint['block_size']
    
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda L: "".join([itos[i] for i in L])
    
    # Instantiate the model architecture
    model = MicronGPT(
        n_embd=checkpoint['n_embd'],
        n_head=checkpoint['n_head'],
        n_layer=checkpoint['n_layer'],
        vocab_size=vocab_size,
        block_size=block_size_chk,
        dropout=0.0
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("\n==================================================")
    print("      Welcome to Micron Chat (RTX 3050 Edition)   ")
    print("  Trained locally. Ask me any weird questions!    ")
    print("  Type 'exit' or 'quit' to end the session.        ")
    print("==================================================\n")
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.strip().lower() in ['exit', 'quit']:
                print("Micron: Goodbye! Keep the GPU cool!")
                break
                
            if not user_input.strip():
                continue
                
            # Format context prompt
            prompt = f"Question: {user_input.strip()}\nAnswer:"
            context_tokens = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
            
            print("Micron: ", end="", flush=True)
            
            input_seq = context_tokens
            response_buffer = ""
            
            # Generate characters sequentially (streaming effect)
            for _ in range(500): # max response characters
                idx_cond = input_seq[:, -block_size_chk:]
                
                with torch.no_grad():
                    if device == 'cuda':
                        with torch.amp.autocast('cuda'):
                            logits, _ = model(idx_cond)
                    else:
                        logits, _ = model(idx_cond)
                        
                logits = logits[:, -1, :]
                
                # Apply gentle repetition penalty for recently generated characters (excluding spacing)
                recent_tokens = input_seq[0, -30:].tolist()
                for t in set(recent_tokens):
                    char = decode([t])
                    if char.isalnum():
                        count = recent_tokens.count(t)
                        logits[0, t] -= (0.45 * count) # decrease probability of repeating chars
                        
                # Use a temperature of 0.5 and top_k=5
                logits = logits / 0.5
                v, _ = torch.topk(logits, min(5, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
                
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
                input_seq = torch.cat((input_seq, idx_next), dim=1)
                char_new = decode([idx_next.item()])
                
                response_buffer += char_new
                
                # Check for prompt separation delimiter
                if response_buffer.endswith("***") or response_buffer.endswith("\n\nQuestion:"):
                    # Strip delimiter and break
                    if response_buffer.endswith("***"):
                        response_buffer = response_buffer[:-3]
                    elif response_buffer.endswith("\n\nQuestion:"):
                        response_buffer = response_buffer[:-11]
                    break
                    
                print(char_new, end="", flush=True)
                
            print("\n")
            
        except KeyboardInterrupt:
            print("\nMicron: Goodbye! Terminating...")
            break

# -----------------
# Main Entry Point
# -----------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Micron: Upgraded GPT model with save/load and chat interface")
    parser.add_argument("--train", action="store_true", help="Train the model on the dataset")
    parser.add_argument("--resume", action="store_true", help="Resume training from an existing checkpoint")
    parser.add_argument("--chat", action="store_true", help="Start local interactive chat session")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/micron_model.pt", help="Path to save/load checkpoint")
    parser.add_argument("--data", type=str, default="data/qa_dataset.txt", help="Path to training dataset file")
    parser.add_argument("--max_iters", type=int, default=None, help="Override default max training iterations")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    
    args = parser.parse_args()
    
    # Overwrite hyperparameters if needed
    if args.max_iters is not None:
        max_iters = args.max_iters
    
    if args.train:
        train_model(args)
    elif args.chat:
        chat_mode(args)
    else:
        parser.print_help()
        print("\nNote: Please specify either --train to train the model, or --chat to run interactive chat.")
        sys.exit(0)
