"""A weight sync must leave out the weights the engines keep from their own load, and only those."""

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from miles.backends.training_utils.weight_update.updater import WeightUpdater

_UPDATER_MODULE = "miles.backends.training_utils.weight_update.updater"

_TRAINED = "model.layers.3.self_attn.q_proj.weight"
_INDEXER = "model.layers.3.self_attn.sparse_indexer_q.weight"
_SSMAX = "model.layers.3.self_attn.ssmax_s"
_BUCKETS = [[_INDEXER, _TRAINED], [_SSMAX]]


def _names(buckets: list[list[tuple[str, torch.Tensor]]]) -> list[list[str]]:
    return [[name for name, _ in bucket] for bucket in buckets]


def _sync(model_name: str) -> tuple[list[list[str]], list[list[str]]]:
    """``(buckets the protocol sent, buckets begin_sync saw)``, by name, for one sync of ``_BUCKETS``."""
    sent, seen_by_begin = [], []

    def begin_sync(weight_version, iter_buckets):
        seen_by_begin.extend(iter_buckets(materialize=True))
        return True

    protocol = SimpleNamespace(
        use_weight_update_session=False,
        needs_base_resync_for_lora=False,
        is_sender=True,
        group_name="test",
        required_placement=None,
        supports_lora=False,
        begin_sync=begin_sync,
        send_bucket=sent.append,
        after_base_weights=lambda: None,
        finalize=lambda weight_version: None,
        after_engines_resumed=lambda: None,
    )
    iterator = SimpleNamespace(
        iter_hf_weights=lambda *args, **kwargs: iter(
            [[(name, torch.zeros(1)) for name in bucket] for bucket in _BUCKETS]
        )
    )
    process_group = SimpleNamespace(get_rank=lambda: 0, barrier=lambda group: None)
    with (
        patch(f"{_UPDATER_MODULE}.get_weight_transfer_protocol", return_value=protocol),
        patch(f"{_UPDATER_MODULE}.dist", process_group),
        patch(f"{_UPDATER_MODULE}.get_gloo_group", return_value=None),
    ):
        WeightUpdater(
            Namespace(check_lora_weight_equal=False),
            [],
            weights_getter=lambda: {},
            model_name=model_name,
            quantization_config=None,
            iterator_factory=lambda *args, **kwargs: iterator,
            parallel_state=None,
            is_lora=False,
        ).update_weights(weight_version=1)
    return _names(sent), _names(seen_by_begin)


def test_step5_sync_leaves_the_indexer_to_the_engines():
    sent, seen_by_begin = _sync("step5")
    # the bucket holding only ssmax_s is not sent at all
    assert sent == seen_by_begin == [[_TRAINED]]


@pytest.mark.parametrize("model_name", ["qwen3moe", "glm4moe"])
def test_other_models_sync_every_weight(model_name):
    sent, seen_by_begin = _sync(model_name)
    assert sent == seen_by_begin == _BUCKETS
