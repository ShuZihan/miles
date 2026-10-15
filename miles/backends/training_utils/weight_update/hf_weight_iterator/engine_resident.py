"""HF weights an sglang loader keeps from its own load, which a weight sync therefore never sends.

The HF weight iterator still yields them, so an HF export, which must hold every weight, writes them.
"""


def is_engine_resident(model_name: str, hf_name: str) -> bool:
    """Step5's DSA indexer: it never trains, and SGLang keeps it through the release of its weights'
    memory, as an update group of its own (``Step5ForCausalLM.end_weight_update``)."""
    return "step5" in model_name.lower() and (
        ".self_attn.sparse_indexer_" in hf_name or hf_name.endswith(".self_attn.ssmax_s")
    )
