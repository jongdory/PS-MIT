from typing import Union

from models.ema import EMA
from models.unet import UNetModel

MODEL_CONFIGS = {
    "brats": {
        "in_channels": 4,
        "model_channels": 128,
        "out_channels": 4,
        "num_res_blocks": 2,
        "attention_resolutions": [8, 16],
        "dropout": 0.0,
        "channel_mult": [1, 2, 4, 4],
        "num_classes": None,
        "use_checkpoint": False,
        "num_heads": 4,
        "num_head_channels": -1,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "resblock_updown": True,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
    "ixi": {
        "in_channels": 3,
        "model_channels": 128,
        "out_channels": 3,
        "num_res_blocks": 2,
        "attention_resolutions": [8, 16],
        "dropout": 0.0,
        "channel_mult": [1, 2, 4, 4],
        "num_classes": None,
        "use_checkpoint": False,
        "num_heads": 4,
        "num_head_channels": -1,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "resblock_updown": True,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
}



def instantiate_model(
    architechture: str, use_ema: bool
) -> UNetModel: 
    assert (
        architechture in MODEL_CONFIGS
    ), f"Model architecture {architechture} is missing its config."

    model = UNetModel(**MODEL_CONFIGS[architechture])

    if use_ema:
        return EMA(model=model)
    else:
        return model
