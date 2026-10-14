"""Qwen3.8-Next PLE (per-layer n-gram embedding) for torchtitan.

The layer hashes each token's trailing n-grams into rows of a frozen table, projects
the gathered rows, and adds a gated, convolved increment to the hyper-connection
residual. Hashing is shared with the Megatron layer and the kernels with its tests,
so the selected rows and the increment match what sglang computed at rollout.

The table is ~102 GB on the full model and is never trained. It is not a parameter
or buffer -- so FSDP, offload, the optimizer and checkpoints never see it -- and it
is held once per node: the first rank to need it writes it to ``/dev/shm``, every
rank on the node maps that one file and registers it with CUDA, and the gather
kernel reads rows straight from host memory. Under FSDP each rank sees different
tokens, so Megatron's row-sharding over TP has nothing to shard across here.
"""

import hashlib
import logging
import os
import time
from dataclasses import dataclass

import torch
from torch import nn
from torchtitan.models.common import Linear
from torchtitan.protocols.module import Module

from miles.kernels.embedding.ple_gate_conv import ple_gate_conv
from miles.kernels.embedding.ple_gather import gather_ple_rows
from miles_plugins.models.qwen3_8_next.ops.ngram import (
    build_ngram_contexts_packed,
    checkpoint_weight_map,
    ngram_hash_ids,
    safetensors_slice,
)

logger = logging.getLogger(__name__)

_SHM_DIR = "/dev/shm"
_TABLE_WAIT_SECONDS = 3600
_HASH_TENSORS = ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets")


class PerLayerEmbedding(Module):
    """``x`` ``[1, L, n*dim]`` + token ids ``[1, L]`` -> ``x`` plus the PLE increment."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        n: int
        embed_dim: int
        hf_layer_index: int
        ngram_size: int
        heads_per_ngram: int
        conv_kernel_size: int
        eos_token_id: int
        key_proj: Linear.Config
        value_proj: Linear.Config
        eps: float = 1e-6
        # the table and hash constants are read from here; the model config fills it in from the trainer's
        hf_checkpoint: str | None = None

    def __init__(self, config: Config):
        super().__init__()
        wide = config.n * config.dim
        self.n = config.n
        self.eps = config.eps
        self.ngram_size = config.ngram_size
        self.heads_per_ngram = config.heads_per_ngram
        self.eos_token_id = config.eos_token_id
        self.hf_layer_index = config.hf_layer_index
        self.hf_checkpoint = config.hf_checkpoint

        self.key_proj = config.key_proj.build()
        self.value_proj = config.value_proj.build()
        self.norm_key = nn.Parameter(torch.empty(wide))
        self.norm_query = nn.Parameter(torch.empty(wide))
        self.norm_conv = nn.Parameter(torch.empty(wide))
        self.conv1d = nn.Parameter(torch.empty(wide, 1, config.conv_kernel_size))

        self._table: torch.Tensor | None = None
        self._hash: dict[str, torch.Tensor] | None = None

    def forward(self, x: torch.Tensor, tokens: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        table, hash_tensors = self._frozen_state(x.device)
        contexts = build_ngram_contexts_packed(tokens.reshape(-1), cu_seqlens, self.ngram_size, self.eos_token_id)
        ids = ngram_hash_ids(
            contexts,
            hash_tensors["layer_multipliers"],
            hash_tensors["ngram_heads_vocab_sizes"],
            hash_tensors["ngram_heads_offsets"],
            self.ngram_size,
            self.heads_per_ngram,
            self.eos_token_id,
        )
        # the table is bf16; match the residual's dtype so the projections run in the model's precision
        embeddings = gather_ple_rows(table, ids, 0, table.shape[0]).flatten(start_dim=-2).to(x.dtype)

        increment = ple_gate_conv(
            x.reshape(-1, x.shape[-1]),
            self.key_proj(embeddings),
            self.value_proj(embeddings),
            self.norm_key,
            self.norm_query,
            self.norm_conv,
            self.conv1d,
            self.n,
            self.eps,
            # sglang dilates the short conv by the n-gram size; the checkpoint has no separate field
            self.ngram_size,
            cu_seqlens,
        )
        return x + increment.view_as(x)

    def _frozen_state(self, device: torch.device) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self._table is None:
            if self.hf_checkpoint is None:
                raise RuntimeError("PLE has no checkpoint to read its table from; a zero table would change logits")
            self._table = node_shared_ple_table(self.hf_checkpoint, self.hf_layer_index)
            self._hash = {
                name: tensor.to(device)
                for name, tensor in read_ple_hash_tensors(self.hf_checkpoint, self.hf_layer_index).items()
            }
        return self._table, self._hash


def _ple_prefix(hf_layer_index: int) -> str:
    return f"model.language_model.layers.{hf_layer_index}.ple.ple_embedding"


def read_ple_hash_tensors(hf_checkpoint: str, hf_layer_index: int) -> dict[str, torch.Tensor]:
    """The three int64 tensors that parameterise the n-gram hash."""
    index = checkpoint_weight_map(hf_checkpoint)
    header_cache: dict = {}
    tensors = {}
    for name in _HASH_TENSORS:
        key = f"{_ple_prefix(hf_layer_index)}.{name}"
        path = os.path.join(hf_checkpoint, index[key])
        start, end, shape = safetensors_slice(path, key, header_cache)
        with open(path, "rb") as f:
            f.seek(start)
            raw = f.read(end - start)
        tensors[name] = torch.frombuffer(bytearray(raw), dtype=torch.int64).reshape(shape).clone()
    return tensors


_NODE_TABLES: dict[str, torch.Tensor] = {}


def node_shared_ple_table(hf_checkpoint: str, hf_layer_index: int) -> torch.Tensor:
    """The layer's whole table, ``[rows, embed_dim / heads]`` bf16, one host copy per node."""
    shards = _table_shards(hf_checkpoint, hf_layer_index)
    rows = sum(shape[0] for _, _, _, shape in shards)
    dim = shards[0][3][1]
    path = _shm_path(hf_checkpoint, hf_layer_index, rows, dim)
    if path in _NODE_TABLES:
        return _NODE_TABLES[path]

    _materialize_once(path, shards, nbytes=rows * dim * 2)
    table = torch.from_file(path, shared=True, size=rows * dim, dtype=torch.bfloat16).view(rows, dim)
    status = torch.cuda.cudart().cudaHostRegister(table.data_ptr(), table.numel() * table.element_size(), 0)
    if int(status) != 0:
        raise RuntimeError(f"cudaHostRegister failed ({status}) for the PLE table at {path}")
    assert table.is_pinned(), "the registered PLE table does not report as pinned"
    _NODE_TABLES[path] = table
    logger.info(
        f"PLE layer {hf_layer_index}: mapped {rows} x {dim} table ({rows * dim * 2 / 2**30:.1f} GiB) from {path}"
    )
    return table


def _table_shards(hf_checkpoint: str, hf_layer_index: int) -> list[tuple[str, int, int, list[int]]]:
    """``(file, start, end, shape)`` of every ``ngram_embedding.shard_<i>`` in shard order."""
    index = checkpoint_weight_map(hf_checkpoint)
    prefix = f"{_ple_prefix(hf_layer_index)}.ngram_embedding.shard_"
    shard_ids = sorted(int(k[len(prefix) :].split(".")[0]) for k in index if k.startswith(prefix))
    if not shard_ids or shard_ids != list(range(len(shard_ids))):
        raise ValueError(f"{hf_checkpoint} has PLE shards {shard_ids[:5]}... for layer {hf_layer_index}, not 0..N-1")
    header_cache: dict = {}
    shards = []
    for shard_id in shard_ids:
        key = f"{prefix}{shard_id}.weight"
        path = os.path.join(hf_checkpoint, index[key])
        start, end, shape = safetensors_slice(path, key, header_cache)
        shards.append((path, start, end, shape))
    return shards


def _shm_path(hf_checkpoint: str, hf_layer_index: int, rows: int, dim: int) -> str:
    key = f"{os.path.realpath(hf_checkpoint)}:{hf_layer_index}:{rows}:{dim}"
    return os.path.join(_SHM_DIR, f"miles-qwen38next-ple-{hashlib.sha1(key.encode()).hexdigest()[:16]}.bf16")


def _materialize_once(path: str, shards: list, *, nbytes: int) -> None:
    """Exactly one process on the node writes the file; the rest wait for it to appear."""
    if os.path.exists(path):
        return
    lock = f"{path}.lock"
    try:
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL))
    except FileExistsError:
        deadline = time.monotonic() + _TABLE_WAIT_SECONDS
        while not os.path.exists(path):
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"waited {_TABLE_WAIT_SECONDS}s for another rank to write {path}; if no job is writing "
                    f"it, a crashed one left {lock} behind -- delete it and retry"
                ) from None
            time.sleep(1)
        return

    free = os.statvfs(_SHM_DIR).f_bavail * os.statvfs(_SHM_DIR).f_frsize
    if free < nbytes:
        os.remove(lock)
        raise RuntimeError(f"{_SHM_DIR} has {free / 2**30:.1f} GiB free; the PLE table needs {nbytes / 2**30:.1f} GiB")
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "wb") as out:
        for shard_path, start, end, _ in shards:
            with open(shard_path, "rb") as f:
                f.seek(start)
                remaining = end - start
                while remaining:
                    chunk = f.read(min(remaining, 1 << 28))
                    out.write(chunk)
                    remaining -= len(chunk)
    os.replace(tmp, path)
    os.remove(lock)
