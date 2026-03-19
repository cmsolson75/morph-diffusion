from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torchaudio
from stable_audio_tools.models.diffusion import ConditionedDiffusionModelWrapper

from .config import Config, MODELS
from .interpolate import slerp_xy


def build_metadata(cfg: Config, prompt: str) -> list[dict]:
    if cfg.model_name == MODELS["small"]:
        return [
            {
                "prompt": prompt,
                "seconds_total": cfg.seconds_total,
            }
        ]

    return [
        {
            "prompt": prompt,
            "seconds_start": 0.0,
            "seconds_total": cfg.seconds_total,
        }
    ]


def load_init_audio(path: str, device: str) -> tuple[int, torch.Tensor]:
    audio, sr = torchaudio.load(path)
    audio = audio.to(device)

    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    return sr, audio


def encode_prompt_conditioning(
    model: ConditionedDiffusionModelWrapper, cfg: Config, prompt: str, device: str
) -> dict[str, Any]:
    metadata = build_metadata(cfg, prompt)
    return model.conditioner(metadata, device=device)


def merge_masks(*masks: torch.Tensor) -> torch.Tensor:
    out = masks[0]
    for m in masks[1:]:
        out = out | m
    return out


def mix_anchor_conditioning(
    cond_a: dict[str, Any],
    cond_b: dict[str, Any],
    cond_c: dict[str, Any],
    cond_d: dict[str, Any],
    x: float,
    y: float,
) -> dict[str, Any]:
    prompt_a, mask_a = cond_a["prompt"]
    prompt_b, mask_b = cond_b["prompt"]
    prompt_c, mask_c = cond_c["prompt"]
    prompt_d, mask_d = cond_d["prompt"]
    mixed_prompt = slerp_xy(prompt_a, prompt_b, prompt_c, prompt_d, x=x, y=y)
    mixed_mask = merge_masks(mask_a, mask_b, mask_c, mask_d)

    mixed_cond = dict(cond_a)
    mixed_cond["prompt"] = (mixed_prompt, mixed_mask)
    return mixed_cond


def maybe_load_init_audio(cfg: Config, device: str) -> tuple[int, torch.Tensor] | None:
    if cfg.init_audio_path is None:
        return None
    path = Path(cfg.init_audio_path)
    if not path.exists():
        raise FileNotFoundError(f"init audio not found: {cfg.init_audio_path}")
    return load_init_audio(cfg.init_audio_path, device=device)
