"""
Visualize the sparsity pattern detected by xattn_estimate.

This script loads cached Q/K tensors, runs xattn_estimate to detect
sparse attention patterns, and generates visualizations.
"""

import os
import sys
import pickle
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xattn.src.Xattention import xattn_estimate
from xattn.threshold.llama_threshold import llama_fuse_8, llama_fuse_16


def load_qk_tensors(seq_len: int, output_dir: str = "output"):
    """Load cached Q/K tensors."""
    query_path = os.path.join(output_dir, f"query_{seq_len}.pkl")
    key_path = os.path.join(output_dir, f"key_{seq_len}.pkl")

    if not os.path.exists(query_path) or not os.path.exists(key_path):
        raise FileNotFoundError(
            f"Q/K tensors not found for seq_len={seq_len}. "
            f"Run eval/efficiency/attention_speedup.py first to generate them."
        )

    with open(query_path, "rb") as f:
        q = pickle.load(f)
    with open(key_path, "rb") as f:
        k = pickle.load(f)

    return q, k


def get_sparsity_mask(q, k, stride: int, threshold, block_size: int = 128):
    """Run xattn_estimate to get sparsity mask."""
    seq_len = q.shape[-2]
    chunk_size = int(
        max(
            min(
                max(2048, 1 << (seq_len - 1).bit_length()),
                128 * 1024 * 2048 // (1 << (seq_len - 1).bit_length()),
            ),
            2048,
        )
    )

    attn_sums, simple_masks = xattn_estimate(
        q, k,
        block_size=block_size,
        stride=stride,
        threshold=threshold,
        select_mode="inverse",
        use_triton=True,
        causal=True,
        chunk_size=chunk_size
    )

    return attn_sums, simple_masks


def visualize_single_head_mask(mask: np.ndarray, head_idx: int, seq_len: int,
                                stride: int, save_path: str = None):
    """Visualize sparsity mask for a single head."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Create custom colormap: white for sparse (False), blue for dense (True)
    cmap = mcolors.ListedColormap(['white', '#4169E1'])

    # Plot the mask
    im = ax.imshow(mask, cmap=cmap, aspect='equal', interpolation='nearest')

    # Add grid lines
    ax.set_xticks(np.arange(-0.5, mask.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, mask.shape[0], 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    # Labels
    ax.set_xlabel('Key Block Index', fontsize=12)
    ax.set_ylabel('Query Block Index', fontsize=12)
    ax.set_title(f'Sparsity Pattern - Head {head_idx}\n'
                 f'Seq Len: {seq_len}, Stride: {stride}, '
                 f'Density: {mask.sum() / mask.size * 100:.1f}%', fontsize=14)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_ticks([0.25, 0.75])
    cbar.set_ticklabels(['Sparse', 'Dense'])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def visualize_all_heads_summary(masks: np.ndarray, seq_len: int, stride: int,
                                 save_path: str = None):
    """Visualize summary of sparsity across all heads."""
    num_heads = masks.shape[1]

    # Calculate density for each head
    densities = [masks[0, h].sum() / masks[0, h].size * 100 for h in range(num_heads)]

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))

    # Subplot 1: Average mask across heads
    ax1 = fig.add_subplot(2, 2, 1)
    avg_mask = masks[0].mean(axis=0)
    im1 = ax1.imshow(avg_mask, cmap='Blues', aspect='equal', vmin=0, vmax=1)
    ax1.set_xlabel('Key Block Index')
    ax1.set_ylabel('Query Block Index')
    ax1.set_title(f'Average Attention Pattern\n(Seq Len: {seq_len}, Stride: {stride})')
    plt.colorbar(im1, ax=ax1, shrink=0.8, label='Density')

    # Subplot 2: Density bar chart
    ax2 = fig.add_subplot(2, 2, 2)
    bars = ax2.bar(range(num_heads), densities, color='#4169E1', alpha=0.7)
    ax2.axhline(y=np.mean(densities), color='red', linestyle='--',
                label=f'Mean: {np.mean(densities):.1f}%')
    ax2.set_xlabel('Head Index')
    ax2.set_ylabel('Density (%)')
    ax2.set_title('Attention Density per Head')
    ax2.legend()
    ax2.set_xticks(range(0, num_heads, max(1, num_heads // 8)))

    # Subplot 3: First head mask
    ax3 = fig.add_subplot(2, 2, 3)
    cmap = mcolors.ListedColormap(['white', '#4169E1'])
    im3 = ax3.imshow(masks[0, 0], cmap=cmap, aspect='equal')
    ax3.set_xlabel('Key Block Index')
    ax3.set_ylabel('Query Block Index')
    ax3.set_title(f'Head 0 Pattern (Density: {densities[0]:.1f}%)')

    # Subplot 4: Head with max density
    ax4 = fig.add_subplot(2, 2, 4)
    max_head = np.argmax(densities)
    im4 = ax4.imshow(masks[0, max_head], cmap=cmap, aspect='equal')
    ax4.set_xlabel('Key Block Index')
    ax4.set_ylabel('Query Block Index')
    ax4.set_title(f'Head {max_head} Pattern (Max Density: {densities[max_head]:.1f}%)')

    plt.suptitle(f'XAttention Sparsity Analysis\nSequence Length: {seq_len}, Stride: {stride}',
                 fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def visualize_multi_length_comparison(seq_lens: list, stride: int, layer_idx: int,
                                       output_dir: str, figures_dir: str):
    """Compare sparsity patterns across different sequence lengths."""
    results = []

    for seq_len in seq_lens:
        try:
            q, k = load_qk_tensors(seq_len, output_dir)
            num_heads = q.shape[1]

            # Get threshold
            threshold_full = torch.tensor(llama_fuse_8 if stride == 8 else llama_fuse_16)[layer_idx]
            threshold = threshold_full[:num_heads] if threshold_full.shape[0] > num_heads else threshold_full

            _, masks = get_sparsity_mask(q, k, stride, threshold)
            masks_np = masks.cpu().numpy()

            # Calculate statistics
            density = masks_np.sum() / masks_np.size * 100
            results.append({
                'seq_len': seq_len,
                'masks': masks_np,
                'density': density,
                'num_blocks': masks_np.shape[-1]
            })
            print(f"Seq {seq_len}: density = {density:.2f}%, blocks = {masks_np.shape[-1]}")

        except FileNotFoundError as e:
            print(f"Skipping seq_len={seq_len}: {e}")
            continue

    if len(results) < 2:
        print("Not enough data for comparison. Run attention_speedup.py first.")
        return

    # Create comparison figure
    fig, axes = plt.subplots(2, len(results), figsize=(5 * len(results), 10))
    if len(results) == 1:
        axes = axes.reshape(2, 1)

    cmap = mcolors.ListedColormap(['white', '#4169E1'])

    for i, res in enumerate(results):
        # Row 1: Head 0 mask
        ax1 = axes[0, i]
        im1 = ax1.imshow(res['masks'][0, 0], cmap=cmap, aspect='equal')
        ax1.set_title(f"{res['seq_len']//1024}K tokens\nDensity: {res['density']:.1f}%")
        ax1.set_xlabel('Key Block')
        ax1.set_ylabel('Query Block')

        # Row 2: Average mask
        ax2 = axes[1, i]
        avg_mask = res['masks'][0].mean(axis=0)
        im2 = ax2.imshow(avg_mask, cmap='Blues', aspect='equal', vmin=0, vmax=1)
        ax2.set_title(f"Average across heads")
        ax2.set_xlabel('Key Block')
        ax2.set_ylabel('Query Block')

    plt.suptitle(f'Sparsity Pattern Comparison (Stride={stride})', fontsize=16, y=1.02)
    plt.tight_layout()

    save_path = os.path.join(figures_dir, f'sparsity_comparison_stride{stride}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

    # Create density trend figure
    fig, ax = plt.subplots(figsize=(10, 6))
    seq_lens_k = [r['seq_len'] // 1024 for r in results]
    densities = [r['density'] for r in results]

    ax.plot(seq_lens_k, densities, 'o-', markersize=10, linewidth=2, color='#4169E1')
    ax.set_xlabel('Sequence Length (K tokens)', fontsize=12)
    ax.set_ylabel('Attention Density (%)', fontsize=12)
    ax.set_title(f'Attention Density vs Sequence Length (Stride={stride})', fontsize=14)
    ax.grid(True, alpha=0.3)

    for i, (x, y) in enumerate(zip(seq_lens_k, densities)):
        ax.annotate(f'{y:.1f}%', (x, y), textcoords="offset points",
                   xytext=(0, 10), ha='center', fontsize=10)

    save_path = os.path.join(figures_dir, f'density_trend_stride{stride}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize XAttention sparsity patterns')
    parser.add_argument('--seq_len', type=int, default=None,
                        help='Sequence length to visualize (e.g., 4096, 8192)')
    parser.add_argument('--stride', type=int, default=16, choices=[8, 16],
                        help='Stride for xattn_estimate')
    parser.add_argument('--layer', type=int, default=12,
                        help='Layer index for threshold')
    parser.add_argument('--head', type=int, default=None,
                        help='Specific head to visualize (default: all heads)')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory containing cached Q/K tensors')
    parser.add_argument('--figures_dir', type=str, default='figures',
                        help='Directory to save figures')
    parser.add_argument('--compare', action='store_true',
                        help='Compare sparsity across multiple sequence lengths')

    args = parser.parse_args()

    os.makedirs(args.figures_dir, exist_ok=True)

    if args.compare:
        # Compare across available sequence lengths
        available_lens = []
        for f in os.listdir(args.output_dir):
            if f.startswith('query_') and f.endswith('.pkl'):
                try:
                    seq_len = int(f.replace('query_', '').replace('.pkl', ''))
                    available_lens.append(seq_len)
                except ValueError:
                    continue

        available_lens.sort()
        print(f"Found cached tensors for: {[l//1024 for l in available_lens]}K")

        visualize_multi_length_comparison(
            available_lens, args.stride, args.layer,
            args.output_dir, args.figures_dir
        )
    else:
        # Single sequence length visualization
        if args.seq_len is None:
            # Try to find available tensors
            for default_len in [4096, 8192, 16384]:
                try:
                    q, k = load_qk_tensors(default_len, args.output_dir)
                    args.seq_len = default_len
                    break
                except FileNotFoundError:
                    continue

            if args.seq_len is None:
                print("No cached Q/K tensors found. Run attention_speedup.py first.")
                return

        print(f"Loading Q/K tensors for seq_len={args.seq_len}...")
        q, k = load_qk_tensors(args.seq_len, args.output_dir)
        print(f"Q shape: {q.shape}, K shape: {k.shape}")

        num_heads = q.shape[1]

        # Get threshold
        threshold_full = torch.tensor(
            llama_fuse_8 if args.stride == 8 else llama_fuse_16
        )[args.layer]
        threshold = threshold_full[:num_heads] if threshold_full.shape[0] > num_heads else threshold_full

        print(f"Running xattn_estimate with stride={args.stride}...")
        attn_sums, masks = get_sparsity_mask(q, k, args.stride, threshold)

        masks_np = masks.cpu().numpy()
        print(f"Mask shape: {masks_np.shape}")
        print(f"Overall density: {masks_np.sum() / masks_np.size * 100:.2f}%")

        # Generate visualizations
        if args.head is not None:
            # Single head
            save_path = os.path.join(
                args.figures_dir,
                f'sparsity_head{args.head}_seq{args.seq_len}_stride{args.stride}.png'
            )
            visualize_single_head_mask(
                masks_np[0, args.head], args.head, args.seq_len,
                args.stride, save_path
            )
        else:
            # All heads summary
            save_path = os.path.join(
                args.figures_dir,
                f'sparsity_summary_seq{args.seq_len}_stride{args.stride}.png'
            )
            visualize_all_heads_summary(
                masks_np, args.seq_len, args.stride, save_path
            )

            # Also save individual head visualizations
            for h in range(min(4, num_heads)):  # First 4 heads
                save_path = os.path.join(
                    args.figures_dir,
                    f'sparsity_head{h}_seq{args.seq_len}_stride{args.stride}.png'
                )
                visualize_single_head_mask(
                    masks_np[0, h], h, args.seq_len, args.stride, save_path
                )


if __name__ == '__main__':
    main()
