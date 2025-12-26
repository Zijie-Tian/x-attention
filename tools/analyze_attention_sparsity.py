"""
Analyze real attention score distribution and sparsity masks.

This tool:
1. Loads Q and K tensors from cached files
2. Computes full attention scores: softmax(Q*K^T/sqrt(d))
3. For a given sparsity ratio, finds the mask covering top attention values
4. Visualizes both the real attention distribution and the selected mask
"""

import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math

#############################################
# Configuration
#############################################

# Sequence lengths to analyze (in tokens)
# For seq_len > 4096, only block-level analysis is performed (token-level is too expensive)
SEQ_LENS = [4096, 8192, 16384, 32768, 65536, 131072]

MASS_COVERAGES = [0.9, 0.95, 0.99]  # Target attention mass coverage values to test
HEAD_IDX = 0             # Which head to visualize (None for all)
BLOCK_SIZE = 128         # Block size for block-level analysis
OUTPUT_DIR = "output"    # Directory containing cached Q/K tensors
FIGURES_DIR = "figures/analysis"  # Directory to save figures

# Threshold for token-level analysis (skip for longer sequences)
TOKEN_LEVEL_MAX_SEQ = 4096


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


def compute_attention_scores(q, k, head_idx=0, causal=True):
    """
    Compute full attention scores for a single head.

    Args:
        q: Query tensor [batch, heads, seq_len, head_dim]
        k: Key tensor [batch, heads, seq_len, head_dim]
        head_idx: Which head to compute
        causal: Whether to apply causal masking

    Returns:
        attention_scores: [seq_len, seq_len] after softmax
        raw_scores: [seq_len, seq_len] before softmax (scaled QK^T)
    """
    batch, num_heads, seq_len, head_dim = q.shape
    scale = 1.0 / math.sqrt(head_dim)

    # Extract single head
    q_h = q[0, head_idx]  # [seq_len, head_dim]
    k_h = k[0, head_idx]  # [seq_len, head_dim]

    print(f"Computing attention scores for head {head_idx}...")
    print(f"  Sequence length: {seq_len}")
    print(f"  Head dimension: {head_dim}")
    print(f"  Scale: {scale:.6f}")

    # Compute QK^T / sqrt(d)
    raw_scores = torch.matmul(q_h, k_h.T) * scale  # [seq_len, seq_len]

    # Apply causal mask
    if causal:
        causal_mask = torch.triu(torch.ones_like(raw_scores, dtype=torch.bool), diagonal=1)
        raw_scores = raw_scores.masked_fill(causal_mask, float('-inf'))

    # Compute softmax (row-wise)
    attention_scores = torch.softmax(raw_scores, dim=-1)

    return attention_scores, raw_scores


def compute_block_attention_scores(attention_scores, block_size=128):
    """
    Aggregate attention scores at block level.

    Args:
        attention_scores: [seq_len, seq_len]
        block_size: Size of each block

    Returns:
        block_scores: [num_blocks, num_blocks] - sum of attention in each block
    """
    seq_len = attention_scores.shape[0]
    num_blocks = seq_len // block_size

    block_scores = torch.zeros(num_blocks, num_blocks, device=attention_scores.device)

    for i in range(num_blocks):
        for j in range(num_blocks):
            q_start, q_end = i * block_size, (i + 1) * block_size
            k_start, k_end = j * block_size, (j + 1) * block_size
            block_scores[i, j] = attention_scores[q_start:q_end, k_start:k_end].sum()

    return block_scores


def compute_block_attention_scores_from_qk(q, k, head_idx=0, block_size=128, causal=True):
    """
    Compute block-level attention scores directly from Q and K tensors.
    This avoids materializing the full seq_len x seq_len attention matrix.

    For each block (i, j), computes the sum of softmax attention weights.

    Args:
        q: Query tensor [batch, heads, seq_len, head_dim]
        k: Key tensor [batch, heads, seq_len, head_dim]
        head_idx: Which head to compute
        block_size: Size of each block
        causal: Whether to apply causal masking

    Returns:
        block_scores: [num_blocks, num_blocks] - sum of attention in each block
    """
    batch, num_heads, seq_len, head_dim = q.shape
    scale = 1.0 / math.sqrt(head_dim)
    num_blocks = seq_len // block_size

    # Extract single head
    q_h = q[0, head_idx]  # [seq_len, head_dim]
    k_h = k[0, head_idx]  # [seq_len, head_dim]

    print(f"Computing block attention scores directly from Q/K...")
    print(f"  Sequence length: {seq_len}")
    print(f"  Num blocks: {num_blocks}")
    print(f"  Block size: {block_size}")

    block_scores = torch.zeros(num_blocks, num_blocks, device=q.device)

    # Process one query block row at a time to save memory
    for i in range(num_blocks):
        q_start, q_end = i * block_size, (i + 1) * block_size
        q_block = q_h[q_start:q_end]  # [block_size, head_dim]

        # Compute attention scores for this query block against all key blocks
        # First compute raw scores for the entire row
        raw_scores = torch.matmul(q_block, k_h.T) * scale  # [block_size, seq_len]

        # Apply causal mask
        if causal:
            # Create causal mask for this query block
            # Query positions: [q_start, q_end), Key positions: [0, seq_len)
            q_positions = torch.arange(q_start, q_end, device=q.device).unsqueeze(1)  # [block_size, 1]
            k_positions = torch.arange(seq_len, device=q.device).unsqueeze(0)  # [1, seq_len]
            causal_mask = k_positions > q_positions  # [block_size, seq_len]
            raw_scores = raw_scores.masked_fill(causal_mask, float('-inf'))

        # Compute softmax for the entire row
        attention_row = torch.softmax(raw_scores, dim=-1)  # [block_size, seq_len]

        # Sum attention into blocks
        for j in range(num_blocks):
            k_start, k_end = j * block_size, (j + 1) * block_size
            block_scores[i, j] = attention_row[:, k_start:k_end].sum()

        if (i + 1) % 64 == 0 or i == num_blocks - 1:
            print(f"  Processed query block {i+1}/{num_blocks}")

    return block_scores


def compute_sparsity_mask_by_mass(attention_scores, mass_coverage=0.9):
    """
    Compute a sparsity mask that covers the target fraction of attention mass.

    For each row, we select the minimum number of positions that cover
    mass_coverage fraction of the attention weight for that row.

    Args:
        attention_scores: [seq_len, seq_len] softmax attention
        mass_coverage: Target attention mass to cover (0.9 = keep positions covering 90% of mass)

    Returns:
        mask: [seq_len, seq_len] - True = keep, False = sparse
        actual_sparsity: Actual achieved sparsity ratio
        attention_mass_covered: Actual fraction of attention mass covered by mask
    """
    seq_len = attention_scores.shape[0]

    # Per-row mass-based: for each row, keep minimum positions to cover mass_coverage of mass
    mask = torch.zeros_like(attention_scores, dtype=torch.bool)

    for i in range(seq_len):
        row = attention_scores[i]

        # Sort indices by attention value (descending)
        sorted_indices = torch.argsort(row, descending=True)
        sorted_values = row[sorted_indices]

        # Accumulate until we reach target mass
        cumsum = torch.cumsum(sorted_values, dim=0)

        # Find how many positions we need to cover mass_coverage
        num_to_keep = (cumsum < mass_coverage).sum().item() + 1

        # Mark top positions as kept
        keep_indices = sorted_indices[:num_to_keep]
        mask[i, keep_indices] = True

    # Calculate actual sparsity (against causal positions only)
    causal_positions = seq_len * (seq_len + 1) // 2
    kept_positions = mask.sum().item()
    actual_sparsity = 1.0 - kept_positions / causal_positions

    # Calculate attention mass covered
    attention_mass_covered = (attention_scores * mask.float()).sum().item() / attention_scores.sum().item()

    return mask, actual_sparsity, attention_mass_covered


def compute_block_sparsity_mask_by_mass(block_scores, mass_coverage=0.9, causal=True):
    """
    Compute block-level sparsity mask that covers target fraction of attention mass.

    For each query block row, select blocks that cover mass_coverage fraction of attention.

    Args:
        block_scores: [num_blocks, num_blocks]
        mass_coverage: Target attention mass to cover (0.9 = cover 90% of mass)
        causal: Whether causal masking is applied

    Returns:
        mask: [num_blocks, num_blocks] - True = keep, False = sparse
        actual_sparsity: Actual achieved sparsity
        attention_mass_covered: Actual fraction of attention mass covered
    """
    num_blocks = block_scores.shape[0]

    mask = torch.zeros_like(block_scores, dtype=torch.bool)

    for i in range(num_blocks):
        row = block_scores[i].clone()

        # For causal, future blocks have 0 attention
        if causal:
            row[i+1:] = 0

        # Normalize row to get distribution
        row_sum = row.sum()
        if row_sum > 0:
            row_norm = row / row_sum
        else:
            continue

        # Sort indices by attention value (descending)
        sorted_indices = torch.argsort(row_norm, descending=True)
        sorted_values = row_norm[sorted_indices]

        # Accumulate until we reach target mass
        cumsum = torch.cumsum(sorted_values, dim=0)

        # Find how many positions we need
        num_to_keep = (cumsum < mass_coverage).sum().item() + 1

        # Mark top positions as kept
        keep_indices = sorted_indices[:num_to_keep]
        mask[i, keep_indices] = True

    # Count sparsity (excluding future blocks for causal)
    if causal:
        total_valid = sum(i + 1 for i in range(num_blocks))
        kept = mask.sum().item()
        actual_sparsity = 1.0 - kept / total_valid
    else:
        actual_sparsity = 1.0 - mask.float().sum().item() / (num_blocks * num_blocks)

    # Calculate attention mass covered
    total_attention = block_scores.sum().item()
    if total_attention > 0:
        attention_mass_covered = (block_scores * mask.float()).sum().item() / total_attention
    else:
        attention_mass_covered = 0.0

    return mask, actual_sparsity, attention_mass_covered


def visualize_attention_and_mask(attention_scores, mask, seq_len, mass_coverage,
                                  head_idx, figures_dir, actual_sparsity, attention_mass_covered):
    """
    Visualize the real attention distribution and the selected mask.

    Args:
        attention_scores: [seq_len, seq_len] softmax attention
        mask: [seq_len, seq_len] - True = keep
        seq_len: Sequence length
        mass_coverage: Target mass coverage
        head_idx: Head index
        figures_dir: Output directory
        actual_sparsity: Actual achieved sparsity
        attention_mass_covered: Actual fraction of attention mass covered
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    attn_np = attention_scores.cpu().float().numpy()
    mask_np = mask.cpu().numpy().astype(float)

    # Real attention distribution (log scale for visibility)
    ax = axes[0]
    attn_log = np.log10(attn_np + 1e-10)  # Add small epsilon to avoid log(0)
    im0 = ax.imshow(attn_log, cmap='GnBu', aspect='auto', interpolation='nearest')
    ax.set_title(f'Real Attention Distribution (log10 scale)\nHead {head_idx}, Seq Len {seq_len}')
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    plt.colorbar(im0, ax=ax, label='log10(attention)')

    # Sparsity mask with blue-green colormap
    ax = axes[1]
    # Create custom blue-green colormap: blue for sparse (0), green for keep (1)
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#1f77b4', '#2ca02c']  # Blue to Green
    cmap_bg = LinearSegmentedColormap.from_list('BlueGreen', colors)

    im1 = ax.imshow(mask_np, cmap=cmap_bg, aspect='auto',
                    interpolation='nearest', vmin=0, vmax=1)
    ax.set_title(f'Sparsity Mask (Target Mass: {mass_coverage*100:.0f}%)\n'
                 f'Sparsity: {actual_sparsity*100:.1f}%, Mass Covered: {attention_mass_covered*100:.1f}%')
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    cbar = plt.colorbar(im1, ax=ax)
    cbar.set_ticks([0.25, 0.75])
    cbar.set_ticklabels(['Sparse', 'Keep'])

    plt.tight_layout()

    # Save figure
    save_path = Path(figures_dir) / f'attention_sparsity_head{head_idx}_seq{seq_len}_mass{mass_coverage}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {save_path}")
    plt.close()


def visualize_block_attention_and_mask(block_scores, block_mask, seq_len, block_size,
                                         mass_coverage, head_idx, figures_dir, actual_sparsity,
                                         attention_mass_covered):
    """
    Visualize block-level attention and mask.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    block_np = block_scores.cpu().float().numpy()
    mask_np = block_mask.cpu().numpy().astype(float)

    num_blocks = block_np.shape[0]

    # Block attention scores (log scale)
    ax = axes[0]
    block_log = np.log10(block_np + 1e-10)
    # Mask out future blocks for visualization
    for i in range(num_blocks):
        block_log[i, i+1:] = np.nan
    im0 = ax.imshow(block_log, cmap='GnBu', aspect='auto', interpolation='nearest')
    ax.set_title(f'Block Attention Distribution (log10 scale)\n'
                 f'Head {head_idx}, {num_blocks}x{num_blocks} blocks (size={block_size})')
    ax.set_xlabel('Key Block')
    ax.set_ylabel('Query Block')
    plt.colorbar(im0, ax=ax, label='log10(attention sum)')

    # Block mask with blue-green colormap
    ax = axes[1]
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#1f77b4', '#808080', '#2ca02c']  # Blue, Gray, Green
    cmap_bg = LinearSegmentedColormap.from_list('BlueGrayGreen', colors)

    mask_vis = mask_np.copy()
    for i in range(num_blocks):
        mask_vis[i, i+1:] = 0.5  # Mark future as gray
    im1 = ax.imshow(mask_vis, cmap=cmap_bg, aspect='auto',
                    interpolation='nearest', vmin=0, vmax=1)
    ax.set_title(f'Block Sparsity Mask (Target Mass: {mass_coverage*100:.0f}%)\n'
                 f'Sparsity: {actual_sparsity*100:.1f}%, Mass Covered: {attention_mass_covered*100:.1f}%')
    ax.set_xlabel('Key Block')
    ax.set_ylabel('Query Block')
    cbar = plt.colorbar(im1, ax=ax)
    cbar.set_ticks([0.17, 0.5, 0.83])
    cbar.set_ticklabels(['Sparse', 'Future', 'Keep'])

    plt.tight_layout()

    # Save figure
    save_path = Path(figures_dir) / f'block_attention_sparsity_head{head_idx}_seq{seq_len}_bs{block_size}_mass{mass_coverage}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {save_path}")
    plt.close()


def print_statistics(attention_scores, mask, block_scores, block_mask, head_idx,
                     token_mass_covered, block_mass_covered, mass_coverage):
    """Print detailed statistics about the attention and sparsity."""
    print("\n" + "="*70)
    print(f"STATISTICS FOR HEAD {head_idx} (target mass coverage: {mass_coverage*100:.0f}%)")
    print("="*70)

    # Token-level stats
    seq_len = attention_scores.shape[0]
    total_positions = seq_len * seq_len
    causal_positions = seq_len * (seq_len + 1) // 2  # Lower triangular

    kept_positions = mask.sum().item()

    print(f"\nToken-level Analysis:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Total positions: {total_positions}")
    print(f"  Causal positions: {causal_positions}")
    print(f"  Kept positions: {kept_positions}")
    print(f"  Density: {100*kept_positions/causal_positions:.1f}%")
    print(f"  Sparsity: {100*(1 - kept_positions/causal_positions):.1f}%")
    print(f"  Attention mass covered: {100*token_mass_covered:.2f}%")

    # Block-level stats
    num_blocks = block_scores.shape[0]
    causal_blocks = num_blocks * (num_blocks + 1) // 2
    kept_blocks = block_mask.sum().item()

    print(f"\nBlock-level Analysis:")
    print(f"  Number of blocks: {num_blocks}")
    print(f"  Causal blocks: {causal_blocks}")
    print(f"  Kept blocks: {kept_blocks}")
    print(f"  Density: {100*kept_blocks/causal_blocks:.1f}%")
    print(f"  Sparsity: {100*(1 - kept_blocks/causal_blocks):.1f}%")
    print(f"  Attention mass covered: {100*block_mass_covered:.2f}%")

    # Per-row statistics
    print(f"\nPer-row kept positions (sample):")
    for i in [0, seq_len//4, seq_len//2, 3*seq_len//4, seq_len-1]:
        row_kept = mask[i].sum().item()
        row_max = i + 1  # Causal: can only attend to positions 0..i
        print(f"  Row {i:4d}: {row_kept:4d} / {row_max:4d} ({100*row_kept/row_max:.1f}%)")


def main():
    # Create output directory
    figures_path = Path(FIGURES_DIR)
    figures_path.mkdir(parents=True, exist_ok=True)

    print(f"Configuration:")
    print(f"  Sequence lengths: {SEQ_LENS}")
    print(f"  Mass coverage targets: {MASS_COVERAGES}")
    print(f"  Head index: {HEAD_IDX}")
    print(f"  Block size: {BLOCK_SIZE}")
    print(f"  Token-level analysis max seq: {TOKEN_LEVEL_MAX_SEQ}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Store all results for final summary
    all_results = []

    # Process each sequence length
    for seq_len in SEQ_LENS:
        print("\n" + "#"*80)
        print(f"# SEQUENCE LENGTH: {seq_len} ({seq_len//1024}K)")
        print("#"*80)

        # Check if data file exists
        query_path = Path(OUTPUT_DIR) / f"query_{seq_len}.pkl"
        if not query_path.exists():
            print(f"  Skipping - data file not found: {query_path}")
            continue

        # Load data
        q, k = load_qk_data(OUTPUT_DIR, seq_len)
        q = q.to(device)
        k = k.to(device)

        # Determine if we should do token-level analysis
        do_token_level = seq_len <= TOKEN_LEVEL_MAX_SEQ

        if do_token_level:
            # Compute full attention scores for short sequences
            attention_scores, raw_scores = compute_attention_scores(q, k, head_idx=HEAD_IDX, causal=True)

            # Compute block-level from full attention
            print(f"\nComputing block-level attention from full matrix...")
            block_scores = compute_block_attention_scores(attention_scores, block_size=BLOCK_SIZE)
        else:
            # For long sequences, compute block scores directly
            print(f"\nSkipping token-level analysis (seq_len > {TOKEN_LEVEL_MAX_SEQ})")
            attention_scores = None
            block_scores = compute_block_attention_scores_from_qk(
                q, k, head_idx=HEAD_IDX, block_size=BLOCK_SIZE, causal=True
            )

        # Test each mass coverage value
        for mass_coverage in MASS_COVERAGES:
            print("\n" + "="*70)
            print(f"TESTING: seq_len={seq_len}, mass_coverage={mass_coverage*100:.0f}%")
            print("="*70)

            token_sparsity = None
            token_mass = None
            token_mask = None

            if do_token_level:
                # Compute token-level sparsity mask
                print(f"\nComputing token-level sparsity mask...")
                token_mask, token_sparsity, token_mass = compute_sparsity_mask_by_mass(
                    attention_scores, mass_coverage=mass_coverage
                )
                print(f"  Token-level sparsity: {token_sparsity*100:.1f}%")
                print(f"  Token-level mass covered: {token_mass*100:.2f}%")

            # Compute block-level sparsity mask
            print(f"Computing block-level sparsity mask...")
            block_mask, block_sparsity, block_mass = compute_block_sparsity_mask_by_mass(
                block_scores, mass_coverage=mass_coverage, causal=True
            )
            print(f"  Block-level sparsity: {block_sparsity*100:.1f}%")
            print(f"  Block-level mass covered: {block_mass*100:.2f}%")

            # Store results
            all_results.append({
                'seq_len': seq_len,
                'mass_coverage': mass_coverage,
                'token_sparsity': token_sparsity,
                'token_mass': token_mass,
                'block_sparsity': block_sparsity,
                'block_mass': block_mass
            })

            # Visualize token-level (only for short sequences)
            if do_token_level:
                print(f"\nGenerating token-level visualization...")
                visualize_attention_and_mask(
                    attention_scores, token_mask, seq_len, mass_coverage,
                    HEAD_IDX, FIGURES_DIR, token_sparsity, token_mass
                )

            # Visualize block-level
            print(f"Generating block-level visualization...")
            visualize_block_attention_and_mask(
                block_scores, block_mask, seq_len, BLOCK_SIZE,
                mass_coverage, HEAD_IDX, FIGURES_DIR, block_sparsity, block_mass
            )

        # Free memory
        del q, k, block_scores
        if attention_scores is not None:
            del attention_scores
        torch.cuda.empty_cache()

    # Print final summary table
    print("\n" + "="*90)
    print("FINAL SUMMARY TABLE")
    print("="*90)
    print(f"\n{'Seq Len':<12} {'Mass Target':<12} {'Token Sparsity':<16} {'Token Density':<14} {'Block Sparsity':<16} {'Block Density'}")
    print("-"*90)
    for r in all_results:
        seq_str = f"{r['seq_len']//1024}K"
        mass_str = f"{r['mass_coverage']*100:.0f}%"
        if r['token_sparsity'] is not None:
            token_sp = f"{r['token_sparsity']*100:.1f}%"
            token_dn = f"{(1-r['token_sparsity'])*100:.1f}%"
        else:
            token_sp = "N/A"
            token_dn = "N/A"
        block_sp = f"{r['block_sparsity']*100:.1f}%"
        block_dn = f"{(1-r['block_sparsity'])*100:.1f}%"
        print(f"{seq_str:<12} {mass_str:<12} {token_sp:<16} {token_dn:<14} {block_sp:<16} {block_dn}")

    print("\n" + "="*90)
    print("DONE")
    print("="*90)


if __name__ == '__main__':
    main()
