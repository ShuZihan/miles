"""Step5 trainer-side expansion: the engine-expanded prompt swapped in, image positions, idempotence, text samples."""

import pytest
import torch

from miles.backends.training_utils.data.multimodal import (
    STEP5_EXPANDED_PROMPT_KEY,
    STEP5_MODEL_INPUT_KEYS,
    expand_multimodal_rollout_data_in_place,
)
from miles.backends.training_utils.data.rollout import DataIterator, get_batch
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state

_IMAGE = 7
_BOS = 0


def _set_parallel_state(cp_size: int) -> None:
    trivial = GroupInfo(rank=0, size=1, group=None)
    set_parallel_state(
        ParallelState(
            intra_dp=trivial,
            intra_dp_cp=trivial,
            cp=GroupInfo(rank=0, size=cp_size, group=None),
            tp=trivial,
            pp=trivial,
            ep=trivial,
            etp=trivial,
            indep_dp=trivial,
        )
    )


@pytest.fixture(autouse=True)
def _trivial_parallel_state():
    _set_parallel_state(cp_size=1)


def _image_inputs(expanded_prompt: list[int], num_patches: list[int], device: str) -> dict[str, torch.Tensor]:
    expanded = torch.tensor(expanded_prompt, device=device)
    return {
        "step5_pixel_values": torch.randn(len(num_patches), 3, 4, 4, device=device),
        "step5_patch_pixel_values": torch.randn(sum(num_patches), 3, 2, 2, device=device),
        "step5_num_patches": torch.tensor(num_patches, device=device),
        "step5_image_positions": (expanded == _IMAGE).nonzero().flatten(),
        STEP5_EXPANDED_PROMPT_KEY: expanded,
    }


def _rollout_data(device: str = "cpu") -> dict:
    # Request prompts carry one image token per image; the expanded prompts are what SGLang tokenized.
    prompts = [[5, _IMAGE, 8], [5, 6, 8], [9, _IMAGE, _IMAGE]]
    responses = [[20, 21, 22], [23, 24], [25]]
    multimodal_train_inputs = [
        _image_inputs([_BOS, 5, 30, _IMAGE, _IMAGE, _IMAGE, 31, 8], [0], device),
        None,
        _image_inputs([_BOS, 9, 32, _IMAGE, _IMAGE, 33, 30, _IMAGE, 31, 30, _IMAGE, _IMAGE, 31], [1, 0], device),
    ]
    tokens = [torch.tensor(p + r, device=device) for p, r in zip(prompts, responses, strict=True)]
    return {
        "tokens": tokens,
        "loss_masks": [torch.ones(len(r), dtype=torch.int, device=device) for r in responses],
        "total_lengths": [len(t) for t in tokens],
        "response_lengths": [len(r) for r in responses],
        "multimodal_train_inputs": multimodal_train_inputs,
    }


def test_the_expanded_prompt_replaces_the_request_prompt():
    data = _rollout_data()
    expanded_prompts = [
        mm[STEP5_EXPANDED_PROMPT_KEY].tolist() if mm else None for mm in data["multimodal_train_inputs"]
    ]

    expand_multimodal_rollout_data_in_place(data)

    assert data["tokens"][0].tolist() == expanded_prompts[0] + [20, 21, 22]
    assert data["tokens"][1].tolist() == [5, 6, 8, 23, 24]
    assert data["tokens"][2].tolist() == expanded_prompts[2] + [25]
    assert data["total_lengths"] == [len(t) for t in data["tokens"]]
    assert data["response_lengths"] == [3, 2, 1]


def test_image_samples_carry_exactly_the_model_inputs_and_text_samples_none():
    data = _rollout_data()

    expand_multimodal_rollout_data_in_place(data)

    image_inputs, text_inputs, other_image_inputs = data["multimodal_train_inputs"]
    assert tuple(image_inputs) == tuple(other_image_inputs) == STEP5_MODEL_INPUT_KEYS
    assert text_inputs is None


def test_positions_point_at_the_image_tokens():
    data = _rollout_data()

    expand_multimodal_rollout_data_in_place(data)

    for token_tensor, mm in zip(data["tokens"], data["multimodal_train_inputs"], strict=True):
        positions = mm["step5_image_positions"] if mm is not None else torch.empty(0, dtype=torch.long)
        assert (token_tensor[positions] == _IMAGE).all()
        assert positions.numel() == (token_tensor == _IMAGE).sum()


def test_expanding_twice_changes_nothing():
    data = _rollout_data()
    expand_multimodal_rollout_data_in_place(data)
    tokens = [t.clone() for t in data["tokens"]]
    inputs = [dict(mm) if mm is not None else None for mm in data["multimodal_train_inputs"]]

    expand_multimodal_rollout_data_in_place(data)

    assert all(torch.equal(a, b) for a, b in zip(tokens, data["tokens"], strict=True))
    for before, after in zip(inputs, data["multimodal_train_inputs"], strict=True):
        assert (before is None) == (after is None)
        assert before is None or (before.keys() == after.keys() and all(before[k] is after[k] for k in before))


def test_context_parallelism_is_refused():
    _set_parallel_state(cp_size=2)
    with pytest.raises(AssertionError, match="CP>1"):
        expand_multimodal_rollout_data_in_place(_rollout_data())


_BATCH_KEYS = ["tokens", "loss_masks", "total_lengths", "response_lengths", "multimodal_train_inputs"]
_needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="get_batch packs on the current CUDA device")


@_needs_cuda
def test_a_mixed_micro_batch_offsets_positions_into_the_packed_sequence():
    data = _rollout_data(device="cuda")
    expand_multimodal_rollout_data_in_place(data)

    batch = get_batch(DataIterator(data, micro_batch_size=3), _BATCH_KEYS)

    inputs = batch["multimodal_train_inputs"]
    assert tuple(inputs) == STEP5_MODEL_INPUT_KEYS
    assert inputs["step5_image_positions"].numel() == sum(int((t == _IMAGE).sum()) for t in data["tokens"])
    assert (batch["tokens"][0][inputs["step5_image_positions"]] == _IMAGE).all()
    assert inputs["step5_pixel_values"].shape[0] == 3
    assert inputs["step5_num_patches"].tolist() == [0, 1, 0]


@_needs_cuda
def test_a_micro_batch_of_text_samples_hands_the_model_no_image_inputs():
    data = _rollout_data(device="cuda")
    expand_multimodal_rollout_data_in_place(data)

    batch = get_batch(DataIterator(data, micro_batch_indices=[[1], [0, 2]]), _BATCH_KEYS)

    assert batch["multimodal_train_inputs"] == {}
