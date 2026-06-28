import os
import sys
import json
import time
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# Add src folder to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "..", "src")))
from Micron import MicronGPT, device

def evaluate_efficiency():
    print("--------------------------------------------------")
    checkpoint_path = os.path.abspath(os.path.join(script_dir, "..", "checkpoints", "micron_model.pt"))
    dataset_path = os.path.abspath(os.path.join(script_dir, "..", "data", "qa_dataset.txt"))
    output_json_path = os.path.abspath(os.path.join(script_dir, "..", "results", "efficiency_results.json"))

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
    vocab_size = len(chars)
    block_size = checkpoint['block_size']
    n_embd = checkpoint['n_embd']
    n_head = checkpoint['n_head']
    n_layer = checkpoint['n_layer']
    
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    
    # -----------------
    # 1. Analytical FLOP Count Calculation
    # -----------------
    print("Calculating analytical forward-pass FLOPs...")
    # Multi-head Attention per block:
    # - QKV Projection: 3 * (2 * n_embd * n_embd)
    # - Attention Weights (QK^T): 2 * block_size^2 * n_embd
    # - Weighted Aggregation (A @ V): 2 * block_size^2 * n_embd
    # - Attention Output Projection: 2 * n_embd * n_embd
    attn_flops_per_block = 8 * (n_embd ** 2) + 4 * (block_size ** 2) * n_embd
    
    # Feed-forward Network per block:
    # - Up projection: 2 * n_embd * (4 * n_embd) = 8 * n_embd^2
    # - Down projection: 2 * (4 * n_embd) * n_embd = 8 * n_embd^2
    ffn_flops_per_block = 16 * (n_embd ** 2)
    
    # Total for all blocks:
    blocks_flops = n_layer * (attn_flops_per_block + ffn_flops_per_block)
    
    # LM Head:
    # - Projection: 2 * n_embd * vocab_size
    head_flops = 2 * n_embd * vocab_size
    
    # Total FLOPs per token forward pass
    total_flops_per_token = blocks_flops + head_flops
    mflops_per_token = total_flops_per_token / 1e6
    
    print(f"  Attention FLOPs per block: {attn_flops_per_block / 1e6:.2f} MFLOPs")
    print(f"  FFN FLOPs per block:       {ffn_flops_per_block / 1e6:.2f} MFLOPs")
    print(f"  Blocks Total FLOPs:        {blocks_flops / 1e6:.2f} MFLOPs")
    print(f"  LM Head FLOPs:             {head_flops / 1e6:.2f} MFLOPs")
    print(f"  Total FLOPs per token:     {mflops_per_token:.2f} MFLOPs")

    # -----------------
    # 2. Dynamic INT8 Quantization Benchmark (CPU)
    # -----------------
    print("\nRunning CPU dynamic quantization benchmark...")
    # Dynamic quantization only runs on CPU, so we instantiate on CPU
    model_fp32 = MicronGPT(
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        vocab_size=vocab_size,
        block_size=block_size,
        dropout=0.0
    ).to("cpu")
    model_fp32.load_state_dict(checkpoint['model_state_dict'])
    model_fp32.eval()
    
    # Check model sizes in MB
    torch.save(model_fp32.state_dict(), "temp_fp32.pt")
    size_fp32_mb = os.path.getsize("temp_fp32.pt") / (1024 * 1024)
    os.remove("temp_fp32.pt")
    
    # Quantize to INT8
    print("  Quantizing model to INT8...")
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32,
        {nn.Linear},
        dtype=torch.qint8
    )
    
    torch.save(model_int8.state_dict(), "temp_int8.pt")
    size_int8_mb = os.path.getsize("temp_int8.pt") / (1024 * 1024)
    os.remove("temp_int8.pt")
    
    # Benchmark function for throughput (tokens/second)
    # Generate 100 tokens from a standard prompt
    prompt = "Question: Hello Micron, how are you today?\nAnswer:"
    context_tokens = torch.tensor([encode(prompt)], dtype=torch.long, device="cpu")
    
    def benchmark_throughput(model_to_test, description):
        start_time = time.time()
        # Warmup
        with torch.no_grad():
            _ = model_to_test.generate(context_tokens, max_new_tokens=10, temperature=0.6, top_k=5)
            
        start_time = time.time()
        with torch.no_grad():
            output = model_to_test.generate(context_tokens, max_new_tokens=100, temperature=0.6, top_k=5)
        elapsed = time.time() - start_time
        num_generated = 100
        throughput = num_generated / elapsed
        print(f"    {description} Throughput: {throughput:.2f} tokens/sec (Elapsed: {elapsed:.2f}s)")
        return throughput, elapsed
        
    throughput_fp32, time_fp32 = benchmark_throughput(model_fp32, "FP32 Baseline")
    throughput_int8, time_int8 = benchmark_throughput(model_int8, "INT8 Quantized")
    
    speedup = throughput_int8 / throughput_fp32
    print(f"    Quantization Speedup: {speedup:.2f}x")
    print(f"    Size Reduction: FP32 = {size_fp32_mb:.2f} MB | INT8 = {size_int8_mb:.2f} MB")
    
    # Evaluate Validation Loss Delta
    with open(dataset_path, "r", encoding="utf-8") as f:
        text = f.read()
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    val_data = data[n:]
    
    def evaluate_loss(model_to_eval):
        # Sample 40 batches on CPU
        batch_size = 8
        losses = []
        for _ in range(40):
            ix = torch.randint(len(val_data) - block_size, (batch_size,))
            x = torch.stack([val_data[i : i + block_size] for i in ix])
            y = torch.stack([val_data[i + 1 : i + block_size + 1] for i in ix])
            with torch.no_grad():
                _, loss = model_to_eval(x, y)
                losses.append(loss.item())
        return sum(losses) / len(losses)
        
    loss_fp32 = evaluate_loss(model_fp32)
    loss_int8 = evaluate_loss(model_int8)
    
    print(f"    Validation Loss Delta: FP32 Loss = {loss_fp32:.4f} | INT8 Loss = {loss_int8:.4f} (Delta = {loss_int8 - loss_fp32:.4f})")

    # -----------------
    # 3. Training Energy and Cost Analysis
    # -----------------
    print("\nEstimating training energy consumption and cost...")
    training_hours = 20.22
    # Assume 65 Watts average TDP draw for standard multi-core CPU execution
    avg_cpu_wattage = 65.0
    
    energy_kwh = (avg_cpu_wattage * training_hours) / 1000.0
    
    # Electricity rates: average US household cost ~ $0.16 per kWh
    electricity_rate_usd = 0.16
    cost_usd = energy_kwh * electricity_rate_usd
    
    # Carbon Intensity: average US grid emissions factor ~ 0.38 kg CO2e per kWh
    carbon_intensity_factor = 0.38
    co2_emissions_kg = energy_kwh * carbon_intensity_factor
    
    print(f"  Training Time:                {training_hours} hours")
    print(f"  Assumed CPU TDP Wattage:      {avg_cpu_wattage} Watts")
    print(f"  Total Energy Consumed:        {energy_kwh:.3f} kWh")
    print(f"  Estimated Electricity Cost:   ${cost_usd:.2f} (at ${electricity_rate_usd:.2f}/kWh)")
    print(f"  Estimated Carbon Footprint:   {co2_emissions_kg:.3f} kg CO2e")

    # -----------------
    # Save numerical data
    # -----------------
    results = {
        "flops_analysis": {
            "attention_flops_per_block": attn_flops_per_block,
            "ffn_flops_per_block": ffn_flops_per_block,
            "blocks_total_flops": blocks_flops,
            "head_flops": head_flops,
            "total_flops_per_token": total_flops_per_token,
            "mflops_per_token": mflops_per_token
        },
        "quantization_benchmark": {
            "fp32_size_mb": size_fp32_mb,
            "int8_size_mb": size_int8_mb,
            "fp32_throughput_tokens_per_sec": throughput_fp32,
            "int8_throughput_tokens_per_sec": throughput_int8,
            "speedup_factor": speedup,
            "fp32_val_loss": loss_fp32,
            "int8_val_loss": loss_int8,
            "loss_delta": loss_int8 - loss_fp32
        },
        "energy_consumption": {
            "training_hours": training_hours,
            "cpu_wattage_tdp": avg_cpu_wattage,
            "total_energy_kwh": energy_kwh,
            "electricity_cost_usd": cost_usd,
            "carbon_emissions_co2_kg": co2_emissions_kg
        }
    }
    
    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nEfficiency metrics written to: {output_json_path}")
    print("--------------------------------------------------")

if __name__ == "__main__":
    evaluate_efficiency()
