from model_args_utils import load_sibling_model_args


def model_args() -> str:
    # One of each layer kind: dense SWA, MoE full attention, MoE SWA with BF16 experts, dense full attention.
    return load_sibling_model_args(__file__, "step5", layers=(0, 3, 88, 91))
