from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import torch

from .config import Config
from stable_audio_tools.models.diffusion import ConditionedDiffusionModelWrapper

CACHE_ROOT = Path(".cache")
COND_CACHE_DIR = CACHE_ROOT / "cond"
RENDER_CACHE_DIR = CACHE_ROOT / "render"

COND_CACHE_DIR.mkdir(parents=True, exist_ok=True)
RENDER_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def quantize_float(v: float, step: float = 0.01) -> float:
    return round(v / step) * step


def stable_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def init_audio_fingerprint(path: str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    stat = p.stat()
    return f"{p.resolve()}::{stat.st_size}::{stat.st_mtime_ns}"


def cond_cache_key(cfg: Config, prompt: str) -> str:
    return stable_hash(
        {
            "kind": "cond",
            "model_name": cfg.model_name,
            "prompt": prompt,
            "seconds_total": cfg.seconds_total,
        }
    )


def render_cache_key(cfg: Config) -> str:
    return stable_hash(
        {
            "kind": "render",
            "model_name": cfg.model_name,
            "steps": cfg.steps,
            "cfg": cfg.cfg,
            "seconds_total": cfg.seconds_total,
            "x": quantize_float(cfg.x),
            "y": quantize_float(cfg.y),
            "seed": cfg.seed,
            "prompt_a": cfg.prompt_a,
            "prompt_b": cfg.prompt_b,
            "prompt_c": cfg.prompt_c,
            "prompt_d": cfg.prompt_d,
            "init_audio": init_audio_fingerprint(cfg.init_audio_path),
            "init_noise_level": cfg.init_noise_level,
        }
    )


def cond_cache_path(cfg: Config, prompt: str) -> Path:
    return COND_CACHE_DIR / f"{cond_cache_key(cfg, prompt)}.pt"


def render_cache_path(cfg: Config, suffix: str = ".wav") -> Path:
    return RENDER_CACHE_DIR / f"{render_cache_key(cfg)}{suffix}"


def _to_cpu_cond(cond: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in cond.items():
        if isinstance(v, tuple):
            out[k] = tuple(x.detach().cpu() if torch.is_tensor(x) else x for x in v)
        elif isinstance(v, list):
            out[k] = [x.detach().cpu() if torch.is_tensor(x) else x for x in v]
        elif torch.is_tensor(v):
            out[k] = v.detach().cpu()
        else:
            out[k] = v
    return out


def _to_device_cond(cond: dict[str, Any], device: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in cond.items():
        if isinstance(v, tuple):
            out[k] = tuple(x.to(device) if torch.is_tensor(x) else x for x in v)
        elif isinstance(v, list):
            out[k] = [x.to(device) if torch.is_tensor(x) else x for x in v]
        elif torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def get_or_compute_cond(
    model: ConditionedDiffusionModelWrapper,
    cfg: Config,
    prompt: str,
    device: str,
    compute_fn: Callable,
) -> dict[str, Any]:
    path = cond_cache_path(cfg, prompt)
    if path.exists():
        cond_cpu = torch.load(path, map_location="cpu")
        return _to_device_cond(cond_cpu, device)
    cond = compute_fn(model, cfg, prompt, device)
    torch.save(_to_cpu_cond(cond), path)
    return cond
