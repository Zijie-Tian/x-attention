"""Analyze chunk-based vs global vertical-slash estimation overlap."""

import torch
import torch.nn.functional as F
import math
import pickle
from pathlib import Path

#############################################
# Configuration
#############################################
SEQ_LEN = 512
CHUNK_SIZE = 64
HEAD_IDX = 0
OUTPUT_DIR = "output"
VERTICAL_SIZE = 50
SLASH_SIZE = 100

def sum_all_diagonal_matrix(mat):
    b, h, n, m = mat.shape
    zero_mat = torch.zeros((b, h, n, n), device=mat.device, dtype=mat.dtype)
    mat_padded = torch.cat((zero_mat, mat, zero_mat), -1)
    mat_strided = mat_padded.as_strided((1, 1, n, n + m), (1, n * (2 * n + m), 2 * n + m + 1, 1))
    return torch.sum(mat_strided, 2)[:, :, 1:]


def estimate_sparse_pattern(q, k, head_idx, vertical_size, slash_size, q_abs_start):
    """
    Estimate vertical and slash indices.

    Args:
        q: queries [batch, heads, q_len, head_dim]
        k: keys [batch, heads, kv_len, head_dim]
        q_abs_start: absolute starting position of queries (for causal mask)

    Returns:
        vertical_indices: absolute key positions selected
        slash_indices: relative offset from query (how many tokens back)
            slash=0 means attending to key at query position (newest/innermost)
            slash=d means attending to key d positions before query
    """
    q_len, head_dim = q.shape[2], q.shape[3]
    kv_len = k.shape[2]
    vertical_size = min(kv_len, max(vertical_size, 30))
    slash_size = min(kv_len, max(slash_size, 50))

    q_h = q[:, head_idx:head_idx+1, :, :]
    k_h = k[:, head_idx:head_idx+1, :, :]
    last_q = min(64, q_len)

    # Attention: last 64 queries attend to all keys
    # Query i (in last 64) is at absolute position: q_abs_start + q_len - last_q + i
    qk = torch.einsum('bhmk,bhnk->bhmn', q_h[:, :, -last_q:, :], k_h) / math.sqrt(head_dim)

    # Causal mask: query at abs position can only attend to keys <= that position
    # Query i (0 to last_q-1) is at abs position: q_abs_start + q_len - last_q + i
    # Key j is at position j
    # Mask where j > q_abs_start + q_len - last_q + i
    q_abs_positions = q_abs_start + q_len - last_q + torch.arange(last_q, device=q.device)  # [last_q]
    k_positions = torch.arange(kv_len, device=k.device)  # [kv_len]
    causal_mask = k_positions[None, :] <= q_abs_positions[:, None]  # [last_q, kv_len]
    qk = qk.masked_fill(~causal_mask[None, None, :, :], -torch.inf)

    qk = F.softmax(qk, dim=-1, dtype=torch.float32)

    # Vertical: sum across queries -> importance of each key position
    vertical = qk.sum(-2, keepdim=True)
    forced_v = min(5, kv_len // 10)  # Force first 5 or 10% of kv_len
    vertical[..., :forced_v] = torch.inf
    vertical_indices = torch.topk(vertical.squeeze(), vertical_size, -1).indices

    # Slash: sum along diagonals
    # After sum_all_diagonal_matrix: raw index 0 = oldest, raw index kv_len-1 = newest
    slash_sums = sum_all_diagonal_matrix(qk)[..., :-last_q + 1]
    forced_s = min(20, kv_len // 10)  # Force last 20 or 10% of kv_len
    slash_sums[..., -forced_s:] = torch.inf  # Force keep newest diagonals
    slash_raw = torch.topk(slash_sums.squeeze(), slash_size, -1).indices
    # Normalize: slash=0 means newest (0 tokens back), slash=d means d tokens back
    # This makes slash comparable across different context lengths
    slash_indices = (kv_len - 1) - slash_raw

    return vertical_indices, slash_indices


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(Path(OUTPUT_DIR) / f"query_{SEQ_LEN}.pkl", 'rb') as f:
        q = pickle.load(f).to(device)
    with open(Path(OUTPUT_DIR) / f"key_{SEQ_LEN}.pkl", 'rb') as f:
        k = pickle.load(f).to(device)

    num_chunks = SEQ_LEN // CHUNK_SIZE

    # Global estimation: last chunk's queries with all keys
    global_v, global_s = estimate_sparse_pattern(q, k, HEAD_IDX, VERTICAL_SIZE, SLASH_SIZE, q_abs_start=0)
    global_v_set = set(global_v.cpu().tolist())
    global_s_set = set(global_s.cpu().tolist())

    print(f"Seq: {SEQ_LEN}, Chunks: {num_chunks}x{CHUNK_SIZE}, Head: {HEAD_IDX}")
    print(f"Slash indexing: 0=oldest/outermost, kv_len-1=newest/innermost")
    print(f"{'Chunk':<6} {'Q_range':<16} {'K_range':<12} {'V∩G':<6} {'V%':<8} {'S∩G':<6} {'S%':<8}")
    print("-" * 70)

    for chunk_idx in range(num_chunks):
        q_start = chunk_idx * CHUNK_SIZE
        q_end = (chunk_idx + 1) * CHUNK_SIZE
        k_end = (chunk_idx + 1) * CHUNK_SIZE

        q_chunk = q[:, :, q_start:q_end, :]
        k_chunk = k[:, :, :k_end, :]

        chunk_v, chunk_s = estimate_sparse_pattern(
            q_chunk, k_chunk, HEAD_IDX, VERTICAL_SIZE, SLASH_SIZE, q_abs_start=q_start
        )

        chunk_v_set = set(chunk_v.cpu().tolist())
        v_intersect = len(chunk_v_set & global_v_set)

        chunk_s_set = set(chunk_s.cpu().tolist())
        s_intersect = len(chunk_s_set & global_s_set)

        q_range = f"[{q_start}, {q_end})"
        k_range = f"[0, {k_end})"
        print(f"{chunk_idx:<6} {q_range:<16} {k_range:<12} {v_intersect:<6} {v_intersect/len(global_v_set)*100:>5.1f}%   {s_intersect:<6} {s_intersect/len(global_s_set)*100:>5.1f}%")


if __name__ == '__main__':
    main()
