"""Portable PyTorch reference for Step5 selected-region attention and its gradients."""

from __future__ import annotations

import torch
from torch import Tensor

from miles_plugins.models.step5.region_selection import REGION_SIZE, RegionSelection


def sparse_region_attention_torch(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    selection: RegionSelection,
    softmax_scale: float,
) -> Tensor:
    """Compute causal selected-region attention over a THD pack.

    `q` is `[T, H, D]`; `k` and `v` are `[T, G, D]`. Region ids are local to each
    sequence. Invalid slots and future keys are masked; reductions use FP32 and the
    output has `q.dtype`.
    """
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape:
        raise ValueError(f"expected q [T,H,D], k/v [T,G,D], got {q.shape}, {k.shape}, {v.shape}")
    tokens, query_heads, head_dim = q.shape
    _, groups, key_dim = k.shape
    if head_dim != key_dim or query_heads % groups != 0:
        raise ValueError(
            f"query heads must be a multiple of KV groups and dimensions must match; got H={query_heads}, G={groups}, "
            f"q_dim={head_dim}, kv_dim={key_dim}"
        )
    if selection.region_ids.shape[0:2] != (tokens, groups):
        raise ValueError(
            "selection must have one row per token and KV group, "
            f"got {tuple(selection.region_ids.shape)} for q={tuple(q.shape)}"
        )
    if selection.region_ids.dtype != torch.int32:
        raise ValueError(f"selection region ids must be int32, got {selection.region_ids.dtype}")
    if selection.cu_seqlens.ndim != 1 or selection.cu_seqlens[0].item() != 0:
        raise ValueError("selection cu_seqlens must be a 1-D boundary array starting at zero")
    if selection.cu_seqlens[-1].item() != tokens:
        raise ValueError("selection cu_seqlens must cover every query token")

    heads_per_group = query_heads // groups
    output = torch.empty_like(q)
    boundaries = selection.cu_seqlens.tolist()
    offsets = torch.arange(REGION_SIZE, device=q.device)
    for seq_start, seq_end in zip(boundaries[:-1], boundaries[1:], strict=True):
        if seq_end == seq_start:
            continue
        q_seq = q[seq_start:seq_end]
        k_seq = k[seq_start:seq_end]
        v_seq = v[seq_start:seq_end]
        length = seq_end - seq_start
        query_pos = torch.arange(length, device=q.device)[:, None, None]
        seq_output = []
        for group in range(groups):
            region_ids = selection.region_ids[seq_start:seq_end, group]
            region_start = region_ids.clamp_min(0).to(torch.long) * REGION_SIZE
            key_pos = region_start[..., None] + offsets
            valid = (region_ids[..., None] >= 0) & (key_pos < length) & (key_pos <= query_pos)
            gather_pos = key_pos.clamp(0, max(length - 1, 0)).reshape(length, -1)
            selected_k = k_seq[:, group][gather_pos].view(length, -1, head_dim)
            selected_v = v_seq[:, group][gather_pos].view(length, -1, head_dim)

            query = q_seq[:, group * heads_per_group : (group + 1) * heads_per_group].float()
            scores = torch.einsum("lhd,lkd->lhk", query, selected_k.float()) * softmax_scale
            valid_keys = valid.reshape(length, -1)[:, None]
            scores = scores.masked_fill(~valid_keys, float("-inf"))
            # an empty selection returns zero attention, matching the Triton kernel
            probabilities = torch.softmax(scores, dim=-1).masked_fill(~valid_keys, 0.0)
            attended = torch.einsum("lhk,lkd->lhd", probabilities, selected_v.float())
            seq_output.append(attended.to(q.dtype))
        output[seq_start:seq_end] = torch.cat(seq_output, dim=1)
    return output
