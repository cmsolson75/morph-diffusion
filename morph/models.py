from __future__ import annotations

import torch
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.models.diffusion import ConditionedDiffusionModelWrapper
from typing import Tuple

from .config import MODELS


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(
    model_name: str, device: str
) -> Tuple[ConditionedDiffusionModelWrapper, dict]:
    model, model_config = get_pretrained_model(model_name)
    model = model.to(device)

    if device in {"mps", "cpu"}:
        if hasattr(model, "pretransform"):
            model.pretransform.model_half = False
        model = model.to(torch.float32)
    model.eval()
    return model, model_config


def get_sampler_type(model_name: str) -> str:
    return "dpmpp-2m-sde" if model_name == MODELS["main"] else "pingpong"
