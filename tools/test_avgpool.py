"""Test AvgPool_estimate implementation with cached Q/K data."""

import torch
import pickle
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from xattn.src.AvgPool import AvgPool_estimate

# Configuration
SEQ_LEN = 4096
BLOCK_SIZE = 128
TOP_K = 10
OUTPUT_DIR = "output"


def load_qk_data(output_dir: str, seq_len: int):
    """Load Q and K tensors from cached pickle files."""
    with open(Path(output_dir) / f"query_{seq_len}.pkl", 'rb') as f:
        q = pickle.load(f)
    with open(Path(output_dir) / f"key_{seq_len}.pkl", 'rb') as f:
        k = pickle.load(f)
    return q, k


def main():
    # Load data
    q, k = load_qk_data(OUTPUT_DIR, SEQ_LEN)
    print(f"Q: {q.shape}, K: {k.shape}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    q, k = q.to(device), k.to(device)

    num_blocks = SEQ_LEN // BLOCK_SIZE
    print(f"Seq: {SEQ_LEN}, Block size: {BLOCK_SIZE}, Num blocks: {num_blocks}")

    # Test avg pooling
    print("\n--- Testing avg pooling ---")
    block_scores_avg, block_mask_avg = AvgPool_estimate(
        q, k,
        block_size=BLOCK_SIZE,
        top_k=TOP_K,
        causal=True,
        pool_method="avg"
    )
    print(f"Block scores shape: {block_scores_avg.shape}")
    print(f"Block mask shape: {block_mask_avg.shape}")
    print(f"Blocks selected per row: {block_mask_avg[0, 0].sum(dim=-1)[:10].tolist()}...")

    # Test max pooling
    print("\n--- Testing max pooling ---")
    block_scores_max, block_mask_max = AvgPool_estimate(
        q, k,
        block_size=BLOCK_SIZE,
        top_k=TOP_K,
        causal=True,
        pool_method="max"
    )
    print(f"Block scores shape: {block_scores_max.shape}")
    print(f"Block mask shape: {block_mask_max.shape}")

    # Compare avg vs max agreement
    agreement = (block_mask_avg == block_mask_max).float().mean().item() * 100
    print(f"\nAvg vs Max pooling agreement: {agreement:.1f}%")

    # Check causal constraint
    print("\n--- Verifying causal constraint ---")
    causal_violations = 0
    for i in range(num_blocks):
        if block_mask_avg[0, 0, i, i+1:].any():
            causal_violations += 1
    print(f"Causal violations (avg): {causal_violations}")

    for i in range(num_blocks):
        if block_mask_max[0, 0, i, i+1:].any():
            causal_violations += 1
    print(f"Causal violations (max): {causal_violations}")

    # Print sample mask for head 0
    print("\n--- Sample mask (head 0, first 10 rows) ---")
    for i in range(min(10, num_blocks)):
        row = block_mask_avg[0, 0, i].cpu().numpy()
        selected = [j for j, v in enumerate(row) if v]
        print(f"Row {i}: {selected}")

    print("\nTest completed successfully!")


if __name__ == '__main__':
    main()
