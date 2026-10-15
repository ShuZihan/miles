"""Correctness tests for the portable PyTorch Step5 sparse-attention reference."""

import torch

from miles_plugins.models.step5.ops.torch_sparse_attention import sparse_region_attention_torch
from miles_plugins.models.step5.region_selection import REGION_SIZE, RegionSelection


def _selection(lengths: tuple[int, ...], groups: int) -> RegionSelection:
    slots = 3
    rows = []
    for length in lengths:
        for position in range(length):
            current = position // REGION_SIZE
            selected = list(range(min(current + 1, slots)))
            rows.append(selected + [-1] * (slots - len(selected)))
    region_ids = torch.tensor(rows, dtype=torch.int32).view(sum(lengths), 1, slots).expand(-1, groups, -1).clone()
    cu_seqlens = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    return RegionSelection(region_ids=region_ids, cu_seqlens=cu_seqlens)


def _dense_reference(q, k, v, selection, scale):
    output = torch.empty_like(q)
    heads_per_group = q.shape[1] // k.shape[1]
    boundaries = selection.cu_seqlens.tolist()
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        for token in range(start, end):
            for group in range(k.shape[1]):
                regions = selection.region_ids[token, group]
                positions = []
                for region in regions.tolist():
                    if region < 0:
                        continue
                    positions.extend(
                        range(start + region * REGION_SIZE, min(start + (region + 1) * REGION_SIZE, token + 1))
                    )
                key = k[positions, group].float()
                value = v[positions, group].float()
                for head in range(group * heads_per_group, (group + 1) * heads_per_group):
                    weights = torch.softmax(q[token, head].float() @ key.T * scale, dim=-1)
                    output[token, head] = (weights[:, None] * value).sum(0).to(output.dtype)
    return output


def test_torch_reference_matches_explicit_causal_attention():
    torch.manual_seed(0)
    lengths = (11, 9)
    groups, heads, dim = 2, 4, 6
    q = torch.randn(sum(lengths), heads, dim, dtype=torch.bfloat16)
    k = torch.randn(sum(lengths), groups, dim, dtype=torch.bfloat16)
    v = torch.randn(sum(lengths), groups, dim, dtype=torch.bfloat16)
    selection = _selection(lengths, groups)

    actual = sparse_region_attention_torch(q, k, v, selection, dim**-0.5)
    expected = _dense_reference(q, k, v, selection, dim**-0.5)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_torch_reference_propagates_gradients():
    torch.manual_seed(1)
    groups, heads, dim = 2, 4, 8
    selection = _selection((13,), groups)
    q = torch.randn(13, heads, dim, dtype=torch.float32, requires_grad=True)
    k = torch.randn(13, groups, dim, dtype=torch.float32, requires_grad=True)
    v = torch.randn(13, groups, dim, dtype=torch.float32, requires_grad=True)

    sparse_region_attention_torch(q, k, v, selection, dim**-0.5).square().mean().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert k.grad is not None and torch.isfinite(k.grad).all()
    assert v.grad is not None and torch.isfinite(v.grad).all()
