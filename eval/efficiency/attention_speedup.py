try:
    from xattn.src.Xattention import Xattention_prefill
    XATTN_PREFILL = True
except:
    XATTN_PREFILL = False

try:
    from xattn.src.Minference import Minference_prefill
    MINFERENCE_PREFILL = True
except:
    MINFERENCE_PREFILL = False

try:
    from xattn.src.Fullprefill import Full_prefill
    FULL_PREFILL = True
except:
    FULL_PREFILL = False

import pickle
import torch
import time
from generate_prompt import generate_prompt
from xattn.src.load_llama import load_fake_model,FastPrefillConfig
from xattn.threshold.llama_threshold import llama_fuse_8,llama_fuse_16
from transformers import StaticCache
from tqdm import tqdm
import os

if __name__ == "__main__":

    # Sequence lengths in tokens (not K)
    # Examples: 512, 1024, 2048, 4096, 8192
    lens = [512, 1024]

    speedups_xattn_8 = []
    speedups_xattn_16 = []
    speedups_minfer = []
    past_key_values = None
    for seq_len in lens:
        # seq_len is now in tokens directly
        seq_len_display = f"{seq_len//1024}K" if seq_len >= 1024 else f"{seq_len}"
        print(f"Testing {seq_len_display} ({seq_len} tokens)")
        query_path = f"output/query_{seq_len}.pkl"
        key_path = f"output/key_{seq_len}.pkl"
        config = FastPrefillConfig(metric = "xattn",stride = 16)
        layer_to_save = 12
        if not os.path.exists(query_path) or not os.path.exists(key_path):

            model, tokenizer = load_fake_model(name_or_path="/data/models/Llama-3.1-8B-Instruct", layer_to_save=layer_to_save, target_len=seq_len)
            input_ids = generate_prompt(tokenizer, seq_len)
            chunk_size = min(4096, seq_len)  # Use smaller chunk for short sequences
            # Always recreate cache with appropriate size for current sequence length
            past_key_values = StaticCache(config=model.config, max_batch_size=1, max_cache_len=seq_len+1024, device=model.device, dtype=model.dtype)
            with torch.no_grad():
                for i in tqdm(range(0, input_ids.size(1), chunk_size), desc="Prefilling", unit="chunk"):
                    chunk = input_ids[:, i: i + chunk_size]
                    output = model(
                        input_ids=chunk,
                        past_key_values=past_key_values,
                        use_cache=True,
                        num_logits_to_keep=1,
                    )
                    past_key_values = output.past_key_values
            # Release model and cache to free GPU memory for benchmark
            del model, tokenizer, past_key_values, output, input_ids
            torch.cuda.empty_cache()
        with open(query_path, "rb") as f:
            q = pickle.load(f)
        with open(key_path, "rb") as f:
            k = pickle.load(f)
        assert(q.shape[-2] == seq_len)
        assert(k.shape[-2] == seq_len)
        torch.manual_seed(0)

        # Xattention args - adapt threshold to actual head count
        num_heads = q.shape[1]
        threshold_full = torch.tensor(llama_fuse_8)[layer_to_save]
        if threshold_full.shape[0] > num_heads:
            threshold = threshold_full[:num_heads]
        else:
            threshold = threshold_full
        stride = 16
        v = torch.randn(q.shape, dtype=torch.bfloat16).to("cuda").contiguous()
        num_iterations = 50
        num_warmups = 30

        # warm up
        for i in range(num_warmups):
            if XATTN_PREFILL:
                try:
                    Xattention_prefill(q, k, v, stride=16, threshold=threshold, use_triton=True)
                    Xattention_prefill(q, k, v, stride=8, threshold=threshold, use_triton=True)
                except:
                    XATTN_PREFILL = False
            if MINFERENCE_PREFILL:
                try:
                    Minference_prefill(q, k, v)
                except:
                    MINFERENCE_PREFILL = False

        # Efficiency Evaluation
        # For Xattention_prefill stride=8
        total_time_xattn_8 = 0
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start_time = time.time()
            if XATTN_PREFILL:
                try:
                    xattn_output = Xattention_prefill(q, k, v, stride=8, threshold=threshold, use_triton=True, chunk_size=min(32768, seq_len))
                except Exception as e:
                    print(f"Xattention_prefill stride=8 failed: {e}")
                    XATTN_PREFILL = False
            torch.cuda.synchronize()
            total_time_xattn_8 += time.time() - start_time
        avg_time_xattn_8 = total_time_xattn_8 / num_iterations if XATTN_PREFILL else float('inf')

        # For Xattention_prefill stride=16
        total_time_xattn_16 = 0
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start_time = time.time()
            if XATTN_PREFILL:
                try:
                    xattn_output = Xattention_prefill(q, k, v, stride=16, threshold=threshold, use_triton=True, chunk_size=min(32768, seq_len))
                except Exception as e:
                    print(f"Xattention_prefill stride=16 failed: {e}")
                    XATTN_PREFILL = False
            torch.cuda.synchronize()
            total_time_xattn_16 += time.time() - start_time
        avg_time_xattn_16 = total_time_xattn_16 / num_iterations if XATTN_PREFILL else float('inf')

        # For minference
        total_time_minfer = 0
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start_time = time.time()
            if MINFERENCE_PREFILL:
                try:
                    minfer_output = Minference_prefill(q, k, v)
                except Exception as e:
                    print(f"Minference failed: {e}")
                    MINFERENCE_PREFILL = False
            torch.cuda.synchronize()
            total_time_minfer += time.time() - start_time
        avg_time_minfer = total_time_minfer / num_iterations if MINFERENCE_PREFILL else float('inf')

        # For full attention (baseline)
        total_time_full = 0
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start_time = time.time()
            if FULL_PREFILL:
                o = Full_prefill(q, k, v, causal=True)
            torch.cuda.synchronize()
            total_time_full += time.time() - start_time
        avg_time_full = total_time_full / num_iterations

        # Calculate speedups
        print(f"{seq_len_display} Minfer: {avg_time_minfer:.4f}s xattn_8: {avg_time_xattn_8:.4f}s xattn_16: {avg_time_xattn_16:.4f}s full: {avg_time_full:.4f}s")
        speedup_xattn_8 = avg_time_full / avg_time_xattn_8 if avg_time_xattn_8 != float('inf') else 0
        speedup_xattn_16 = avg_time_full / avg_time_xattn_16 if avg_time_xattn_16 != float('inf') else 0
        speedup_minfer = avg_time_full / avg_time_minfer if avg_time_minfer != float('inf') else 0
        speedups_xattn_8.append(speedup_xattn_8)
        speedups_xattn_16.append(speedup_xattn_16)
        speedups_minfer.append(speedup_minfer)

    # Output results
    print(f"\n{'Length':<10}{'Xattn 8 Speedup':<20}{'Xattn 16 Speedup':<25}{'Minfer Speedup'}")
    for seq_len, speedup_xattn_8, speedup_xattn_16, speedup_minfer in zip(lens, speedups_xattn_8, speedups_xattn_16, speedups_minfer):
        seq_len_display = f"{seq_len//1024}K" if seq_len >= 1024 else f"{seq_len}"
        print(f"{seq_len_display:<10}{speedup_xattn_8:<20.2f}{speedup_xattn_16:<25.2f}{speedup_minfer:.2f}")
