"""
Visualize and compare sparsity patterns from two methods:
1. XAttn: xattn_estimate-based sparse detection
2. SpargeAttention: rowmax-based filtering (https://arxiv.org/pdf/2502.18137)

The comparison shows where the two methods agree and disagree on sparsity.
"""

import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import math
import sys

# Import xattn_estimate
try:
    from xattn.src.Xattention import xattn_estimate
except ImportError:
    print("Warning: Could not import xattn_estimate. Make sure xattn is installed.")
    xattn_estimate = None


def load_qk_data(output_dir: str, seq_len: int):
    """Load Q and K tensors from cached files."""
    query_path = Path(output_dir) / f"query_{seq_len}.pkl"
    key_path = Path(output_dir) / f"key_{seq_len}.pkl"

    print(f"Loading Q from {query_path}")
    with open(query_path, 'rb') as f:
        q = pickle.load(f)

    print(f"Loading K from {key_path}")
    with open(key_path, 'rb') as f:
        k = pickle.load(f)

    print(f"Q shape: {q.shape}, K shape: {k.shape}")
    return q, k


def compute_xattn_sparsity(q, k, block_size=128, stride=16, threshold=0.9):
    """
    Compute sparsity mask using xattn_estimate.

    Args:
        q: Query tensor [batch, heads, seq_len, head_dim]
        k: Key tensor [batch, heads, seq_len, head_dim]
        block_size: Block size for tiling
        stride: Stride for xattn_estimate
        threshold: Threshold for xattn_estimate

    Returns:
        sparsity_mask: [batch, heads, num_q_blocks, num_k_blocks] (True = sparse)
    """
    print(f"\nComputing XAttn-based sparsity:")
    print(f"  Block size: {block_size}")
    print(f"  Stride: {stride}")
    print(f"  Threshold: {threshold}")

    # Call xattn_estimate
    batch, num_heads, seq_len, head_dim = q.shape

    attn_sums, simple_masks = xattn_estimate(
        q, k,
        block_size=block_size,
        stride=stride,
        threshold=threshold,
        select_mode="inverse",
        use_triton=True,
        causal=True,
        chunk_size=16384
    )

    # simple_masks: [batch, heads, num_q_blocks_padded, num_k_blocks_padded]
    # Due to chunk_size padding, the mask includes padded blocks
    # We need to extract only the valid region

    print(f"  XAttn full mask shape: {simple_masks.shape}")

    # Calculate actual effective blocks (without padding)
    actual_blocks = seq_len // block_size
    print(f"  Actual effective blocks: {actual_blocks} (seq_len={seq_len}, block_size={block_size})")

    # Extract only the valid region
    simple_masks = simple_masks[:, :, :actual_blocks, :actual_blocks]
    print(f"  XAttn effective mask shape: {simple_masks.shape}")

    # In xattn, True = keep (dense), False = discard (sparse)
    # We need to invert to match rowmax convention (True = sparse)
    sparsity_mask = ~simple_masks

    return sparsity_mask


def compute_rowmax_sparsity(q, k, block_size=128, threshold=5.0, causal=True):
    """
    Compute sparsity mask using rowmax-based filtering.

    Args:
        q: Query tensor [batch, heads, seq_len, head_dim]
        k: Key tensor [batch, heads, seq_len, head_dim]
        block_size: Block size for tiling
        threshold: Threshold for m_local - mij to determine sparsity
        causal: Whether to apply causal masking

    Returns:
        sparsity_mask: [batch, heads, num_q_blocks, num_k_blocks]
        block_rowmax: [batch, heads, num_q_blocks, num_k_blocks]
    """
    batch, num_heads, seq_len, head_dim = q.shape
    num_blocks = math.ceil(seq_len / block_size)

    # Initialize outputs
    sparsity_mask = torch.zeros(batch, num_heads, num_blocks, num_blocks, dtype=torch.bool, device=q.device)
    block_rowmax = torch.zeros(batch, num_heads, num_blocks, num_blocks, device=q.device)

    scale = 1.0 / math.sqrt(head_dim)

    print(f"\nComputing SpargeAttention (rowmax-based) sparsity:")
    print(f"  Block size: {block_size}")
    print(f"  Num blocks: {num_blocks}")
    print(f"  Threshold: {threshold}")

    for b in range(batch):
        for h in range(num_heads):
            q_h = q[b, h]  # [seq_len, head_dim]
            k_h = k[b, h]  # [seq_len, head_dim]

            for i in range(num_blocks):
                q_start = i * block_size
                q_end = min((i + 1) * block_size, seq_len)
                q_block = q_h[q_start:q_end]  # [block_size, head_dim]

                # Compute m_local: max across all previous and current blocks
                m_local = float('-inf')

                for j in range(num_blocks):
                    # Causal mask: skip future blocks
                    if causal and j > i:
                        sparsity_mask[b, h, i, j] = True  # Mark as sparse
                        block_rowmax[b, h, i, j] = float('-inf')
                        continue

                    k_start = j * block_size
                    k_end = min((j + 1) * block_size, seq_len)
                    k_block = k_h[k_start:k_end]  # [block_size, head_dim]

                    # Compute QK^T for this block
                    qk = torch.matmul(q_block, k_block.T) * scale  # [q_block_size, k_block_size]

                    # Apply causal mask within blocks
                    if causal and i == j:
                        mask = torch.triu(torch.ones_like(qk, dtype=torch.bool), diagonal=1)
                        qk = qk.masked_fill(mask, float('-inf'))

                    # Compute rowmax for this block (mij)
                    mij = qk.max().item()
                    block_rowmax[b, h, i, j] = mij

                    # Update m_local
                    m_local = max(m_local, mij)

                # Second pass: determine sparsity based on m_local - mij > threshold
                for j in range(num_blocks):
                    if causal and j > i:
                        continue

                    mij = block_rowmax[b, h, i, j].item()

                    # If the difference is too large, mark as sparse
                    if m_local - mij > threshold:
                        sparsity_mask[b, h, i, j] = True

                if (i + 1) % 10 == 0:
                    print(f"  Head {h}, Query block {i+1}/{num_blocks}")

    print(f"  Rowmax mask shape: {sparsity_mask.shape}")

    return sparsity_mask, block_rowmax


def visualize_comparison_per_head(xattn_mask, rowmax_mask, seq_len, block_size,
                                   xattn_threshold, rowmax_threshold, stride,
                                   figures_dir, head_idx=None):
    """
    Visualize comparison between XAttn and SpargeAttention for each head.

    Args:
        xattn_mask: [batch, heads, num_q_blocks, num_k_blocks] (True = sparse)
        rowmax_mask: [batch, heads, num_q_blocks, num_k_blocks] (True = sparse)
        seq_len: Sequence length
        block_size: Block size
        xattn_threshold: Threshold for xattn
        rowmax_threshold: Threshold for rowmax
        stride: Stride for xattn
        figures_dir: Directory to save figures
        head_idx: If specified, only visualize this head
    """
    batch, num_heads, num_q_blocks, num_k_blocks = xattn_mask.shape

    if head_idx is not None:
        heads_to_plot = [head_idx]
    else:
        heads_to_plot = range(num_heads)

    for h in heads_to_plot:
        xattn = xattn_mask[0, h].cpu().numpy()  # [num_q_blocks, num_k_blocks]
        rowmax = rowmax_mask[0, h].cpu().numpy()

        # Create comparison matrix
        # 0: Both keep (both dense)
        # 1: XAttn keeps, Rowmax discards
        # 2: XAttn discards, Rowmax keeps
        # 3: Both discard (both sparse)
        comparison = np.zeros_like(xattn, dtype=int)
        comparison[~xattn & ~rowmax] = 0  # Both dense
        comparison[~xattn & rowmax] = 1   # XAttn dense, Rowmax sparse
        comparison[xattn & ~rowmax] = 2   # XAttn sparse, Rowmax dense
        comparison[xattn & rowmax] = 3    # Both sparse

        # Calculate statistics
        xattn_density = (~xattn).mean() * 100
        rowmax_density = (~rowmax).mean() * 100
        agreement = ((xattn == rowmax).sum() / xattn.size) * 100

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # XAttn mask
        ax = axes[0]
        xattn_vis = 1 - xattn.astype(float)  # 1 = dense, 0 = sparse
        im0 = ax.imshow(xattn_vis, cmap='RdYlGn', aspect='auto',
                       interpolation='nearest', vmin=0, vmax=1)
        ax.set_title(f'XAttn (stride={stride}, th={xattn_threshold})\nDensity: {xattn_density:.1f}%')
        ax.set_xlabel('Key Blocks')
        ax.set_ylabel('Query Blocks')
        plt.colorbar(im0, ax=ax, label='Dense (1) / Sparse (0)')

        # Rowmax mask
        ax = axes[1]
        rowmax_vis = 1 - rowmax.astype(float)
        im1 = ax.imshow(rowmax_vis, cmap='RdYlGn', aspect='auto',
                       interpolation='nearest', vmin=0, vmax=1)
        ax.set_title(f'SpargeAttention (th={rowmax_threshold})\nDensity: {rowmax_density:.1f}%')
        ax.set_xlabel('Key Blocks')
        ax.set_ylabel('Query Blocks')
        plt.colorbar(im1, ax=ax, label='Dense (1) / Sparse (0)')

        # Comparison
        ax = axes[2]
        # Custom colormap:
        # Blue: Both dense
        # Green: XAttn dense, Rowmax sparse
        # Orange: XAttn sparse, Rowmax dense
        # Red: Both sparse
        colors = ['#2166ac', '#92c5de', '#f4a582', '#b2182b']  # Blue, light blue, light red, red
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(colors)

        im2 = ax.imshow(comparison, cmap=cmap, aspect='auto',
                       interpolation='nearest', vmin=0, vmax=3)
        ax.set_title(f'Comparison (Agreement: {agreement:.1f}%)')
        ax.set_xlabel('Key Blocks')
        ax.set_ylabel('Query Blocks')

        cbar = plt.colorbar(im2, ax=ax, ticks=[0.375, 1.125, 1.875, 2.625])
        cbar.ax.set_yticklabels(['Both Dense', 'XAttn Only', 'Rowmax Only', 'Both Sparse'])

        plt.suptitle(f'Head {h} Sparsity Comparison (Seq={seq_len}, Block={block_size})',
                    fontsize=14, y=1.02)
        plt.tight_layout()

        # Create comparison subdirectory
        comparison_dir = Path(figures_dir) / 'comparison'
        comparison_dir.mkdir(parents=True, exist_ok=True)

        save_path = comparison_dir / f'head{h}_seq{seq_len}_bs{block_size}_s{stride}_xth{xattn_threshold}_rth{rowmax_threshold}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison for head {h} to {save_path}")
        plt.close()


def analyze_comparison_statistics(xattn_mask, rowmax_mask):
    """
    Analyze and print comparison statistics.

    Args:
        xattn_mask: [batch, heads, num_q_blocks, num_k_blocks] (True = sparse)
        rowmax_mask: [batch, heads, num_q_blocks, num_k_blocks] (True = sparse)
    """
    batch, num_heads, num_q_blocks, num_k_blocks = xattn_mask.shape

    print("\n" + "="*80)
    print("COMPARISON STATISTICS")
    print("="*80)

    for h in range(num_heads):
        xattn = xattn_mask[0, h]
        rowmax = rowmax_mask[0, h]

        total_blocks = num_q_blocks * num_k_blocks

        # XAttn statistics
        xattn_sparse = xattn.sum().item()
        xattn_dense = total_blocks - xattn_sparse

        # Rowmax statistics
        rowmax_sparse = rowmax.sum().item()
        rowmax_dense = total_blocks - rowmax_sparse

        # Agreement
        both_dense = (~xattn & ~rowmax).sum().item()
        both_sparse = (xattn & rowmax).sum().item()
        xattn_only = (~xattn & rowmax).sum().item()  # XAttn dense, Rowmax sparse
        rowmax_only = (xattn & ~rowmax).sum().item()  # XAttn sparse, Rowmax dense

        agreement = (both_dense + both_sparse) / total_blocks * 100

        print(f"\nHead {h}:")
        print(f"  XAttn      - Sparse: {xattn_sparse:4d} ({xattn_sparse/total_blocks*100:5.1f}%), Dense: {xattn_dense:4d} ({xattn_dense/total_blocks*100:5.1f}%)")
        print(f"  Rowmax     - Sparse: {rowmax_sparse:4d} ({rowmax_sparse/total_blocks*100:5.1f}%), Dense: {rowmax_dense:4d} ({rowmax_dense/total_blocks*100:5.1f}%)")
        print(f"  Agreement: {agreement:.1f}%")
        print(f"    Both dense:  {both_dense:4d} ({both_dense/total_blocks*100:5.1f}%)")
        print(f"    Both sparse: {both_sparse:4d} ({both_sparse/total_blocks*100:5.1f}%)")
        print(f"    XAttn only:  {xattn_only:4d} ({xattn_only/total_blocks*100:5.1f}%) - XAttn keeps, Rowmax discards")
        print(f"    Rowmax only: {rowmax_only:4d} ({rowmax_only/total_blocks*100:5.1f}%) - Rowmax keeps, XAttn discards")


def main():
    parser = argparse.ArgumentParser(description='Compare XAttn and SpargeAttention sparsity patterns')
    parser.add_argument('--output_dir', type=str, default='output',
                      help='Directory containing cached Q/K tensors')
    parser.add_argument('--seq_len', type=int, default=4096,
                      help='Sequence length to analyze')
    parser.add_argument('--block_size', type=int, default=128,
                      help='Block size for tiling')
    parser.add_argument('--xattn_threshold', type=float, default=0.9,
                      help='Threshold for xattn_estimate')
    parser.add_argument('--rowmax_threshold', type=float, default=5.0,
                      help='Threshold for rowmax method')
    parser.add_argument('--stride', type=int, default=16,
                      help='Stride for xattn_estimate')
    parser.add_argument('--figures_dir', type=str, default='figures',
                      help='Directory to save figures')
    parser.add_argument('--head_idx', type=int, default=None,
                      help='Visualize only this head index (default: all heads)')

    args = parser.parse_args()

    if xattn_estimate is None:
        print("Error: xattn_estimate not available. Please install xattn package.")
        return

    # Create figures directory
    Path(args.figures_dir).mkdir(exist_ok=True)

    # Load data
    q, k = load_qk_data(args.output_dir, args.seq_len)

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    q = q.to(device)
    k = k.to(device)

    # Compute XAttn sparsity
    xattn_mask = compute_xattn_sparsity(
        q, k,
        block_size=args.block_size,
        stride=args.stride,
        threshold=args.xattn_threshold
    )

    # Compute rowmax-based sparsity with same block size
    rowmax_mask, block_rowmax = compute_rowmax_sparsity(
        q, k,
        block_size=args.block_size,
        threshold=args.rowmax_threshold,
        causal=True
    )

    # Analyze comparison
    analyze_comparison_statistics(xattn_mask, rowmax_mask)

    # Visualize comparison
    print(f"\nGenerating comparison visualizations...")
    visualize_comparison_per_head(
        xattn_mask, rowmax_mask,
        seq_len=args.seq_len,
        block_size=args.block_size,
        xattn_threshold=args.xattn_threshold,
        rowmax_threshold=args.rowmax_threshold,
        stride=args.stride,
        figures_dir=args.figures_dir,
        head_idx=args.head_idx
    )

    print("\n" + "="*80)
    print("DONE")
    print("="*80)


if __name__ == '__main__':
    main()
