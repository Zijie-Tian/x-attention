"""
Benchmark script to analyze time breakdown of Xattention_prefill.
Measures xattn_estimate vs block_sparse_attn separately.
"""

import torch
import time
import os
import pickle
from tqdm import tqdm
from transformers import StaticCache

from xattn.src.Xattention import xattn_estimate, Xattention_prefill
from xattn.src.load_llama import load_fake_model, FastPrefillConfig
from xattn.threshold.llama_threshold import llama_fuse_8, llama_fuse_16
from generate_prompt import generate_prompt

try:
    from block_sparse_attn import block_sparse_attn_func
    BLOCK_SPARSE_AVAILABLE = True
except:
    BLOCK_SPARSE_AVAILABLE = False
    print("block_sparse_attn not available")


def benchmark_xattn_estimate(q, k, stride, threshold, block_size=128,
                              num_iterations=50, num_warmups=30, use_triton=True):
    """Benchmark xattn_estimate independently."""
    chunk_size = int(
        max(
            min(
                max(2048, 1 << (q.shape[-2] - 1).bit_length()),
                128 * 1024 * 2048 // (1 << (q.shape[-2] - 1).bit_length()),
            ),
            2048,
        )
    )

    # Warmup
    for _ in range(num_warmups):
        _ = xattn_estimate(
            q, k, block_size=block_size, stride=stride, threshold=threshold,
            select_mode="inverse", use_triton=use_triton, causal=True,
            chunk_size=chunk_size
        )

    # Benchmark
    torch.cuda.synchronize()
    total_time = 0
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start = time.time()
        _ = xattn_estimate(
            q, k, block_size=block_size, stride=stride, threshold=threshold,
            select_mode="inverse", use_triton=use_triton, causal=True,
            chunk_size=chunk_size
        )
        torch.cuda.synchronize()
        total_time += time.time() - start

    return total_time / num_iterations


def benchmark_xattention_prefill(q, k, v, stride, threshold, block_size=128,
                                  num_iterations=50, num_warmups=30, use_triton=True):
    """Benchmark full Xattention_prefill."""
    chunk_size = min(32768, q.shape[-2])

    # Warmup
    for _ in range(num_warmups):
        _ = Xattention_prefill(
            q, k, v, stride=stride, threshold=threshold,
            use_triton=use_triton, chunk_size=chunk_size
        )

    # Benchmark
    torch.cuda.synchronize()
    total_time = 0
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start = time.time()
        _ = Xattention_prefill(
            q, k, v, stride=stride, threshold=threshold,
            use_triton=use_triton, chunk_size=chunk_size
        )
        torch.cuda.synchronize()
        total_time += time.time() - start

    return total_time / num_iterations


if __name__ == "__main__":
    lens = [4, 8, 16, 32, 64, 128]
    stride = 16
    block_size = 128
    layer_to_save = 12
    num_iterations = 50
    num_warmups = 30

    results = []

    for seq_len in lens:
        print(f"\n{'='*60}")
        print(f"Testing {seq_len}K tokens")
        print(f"{'='*60}")

        query_path = f"output/query_{seq_len*1024}.pkl"
        key_path = f"output/key_{seq_len*1024}.pkl"

        # Generate Q/K if not cached
        if not os.path.exists(query_path) or not os.path.exists(key_path):
            config = FastPrefillConfig(metric="xattn", stride=stride)
            model, tokenizer = load_fake_model(
                name_or_path="/home/zijie/models/Llama-3.2-1B-Instruct",
                layer_to_save=layer_to_save,
                target_len=seq_len*1024
            )
            input_ids = generate_prompt(tokenizer, seq_len*1024)
            chunk_size = 4096
            past_key_values = StaticCache(
                config=model.config, batch_size=1,
                max_cache_len=seq_len*1024+1024,
                device=model.device, dtype=model.dtype
            )
            with torch.no_grad():
                for i in tqdm(range(0, input_ids.size(1), chunk_size), desc="Prefilling"):
                    chunk = input_ids[:, i: i + chunk_size]
                    output = model(
                        input_ids=chunk,
                        past_key_values=past_key_values,
                        use_cache=True,
                        num_logits_to_keep=1,
                    )
                    past_key_values = output.past_key_values
            del model, tokenizer, past_key_values, output, input_ids
            torch.cuda.empty_cache()

        # Load cached Q/K
        with open(query_path, "rb") as f:
            q = pickle.load(f)
        with open(key_path, "rb") as f:
            k = pickle.load(f)

        assert q.shape[-2] == seq_len * 1024
        assert k.shape[-2] == seq_len * 1024

        # Prepare threshold
        num_heads = q.shape[1]
        threshold_full = torch.tensor(llama_fuse_8)[layer_to_save]
        threshold = threshold_full[:num_heads] if threshold_full.shape[0] > num_heads else threshold_full

        # Create random V tensor
        v = torch.randn(q.shape, dtype=torch.bfloat16).to("cuda").contiguous()

        # Benchmark xattn_estimate
        print(f"Benchmarking xattn_estimate (stride={stride})...")
        time_estimate = benchmark_xattn_estimate(
            q, k, stride=stride, threshold=threshold, block_size=block_size,
            num_iterations=num_iterations, num_warmups=num_warmups
        )

        # Benchmark full Xattention_prefill
        print(f"Benchmarking Xattention_prefill (stride={stride})...")
        time_prefill = benchmark_xattention_prefill(
            q, k, v, stride=stride, threshold=threshold, block_size=block_size,
            num_iterations=num_iterations, num_warmups=num_warmups
        )

        # Calculate breakdown
        time_sparse_attn = time_prefill - time_estimate
        ratio_estimate = time_estimate / time_prefill * 100
        ratio_sparse = time_sparse_attn / time_prefill * 100

        results.append({
            'seq_len': seq_len,
            'time_estimate': time_estimate,
            'time_prefill': time_prefill,
            'time_sparse_attn': time_sparse_attn,
            'ratio_estimate': ratio_estimate,
            'ratio_sparse': ratio_sparse
        })

        print(f"\nResults for {seq_len}K:")
        print(f"  xattn_estimate:     {time_estimate*1000:.2f} ms ({ratio_estimate:.1f}%)")
        print(f"  block_sparse_attn:  {time_sparse_attn*1000:.2f} ms ({ratio_sparse:.1f}%)")
        print(f"  Total prefill:      {time_prefill*1000:.2f} ms (100%)")

        # Cleanup
        del v
        torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY: Time Breakdown of Xattention_prefill (stride=16)")
    print(f"{'='*80}")
    print(f"{'Length':<10}{'xattn_estimate':<20}{'block_sparse_attn':<20}{'Total':<15}{'Estimate %':<15}")
    print(f"{'-'*80}")
    for r in results:
        print(f"{str(r['seq_len'])+'K':<10}"
              f"{r['time_estimate']*1000:.2f} ms{'':<12}"
              f"{r['time_sparse_attn']*1000:.2f} ms{'':<12}"
              f"{r['time_prefill']*1000:.2f} ms{'':<7}"
              f"{r['ratio_estimate']:.1f}%")
    print(f"{'='*80}")
