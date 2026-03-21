from __future__ import annotations

from pathlib import Path

import torch
from stable_audio_tools.inference.generation import generate_diffusion_cond

from .audio import postprocess_output, save_audio
from .cache import get_or_compute_cond, render_cache_path
from .conditioning import (
    encode_prompt_conditioning,
    maybe_load_init_audio,
    mix_anchor_conditioning,
)
from .config import Config
from .models import get_device, get_sampler_type, load_model


class MorphEngine:
    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.device = device or get_device()
        self.model_name = model_name
        self._loaded_name: str | None = None
        self.model = None
        self.model_config = None

    def _ensure_model(self, model_name: str):
        if self.model is not None and self._loaded_name == model_name:
            return
        self.model, self.model_config = load_model(model_name, self.device)
        self._loaded_name = model_name

    def render_xy(self, cfg: Config) -> Path:
        self._ensure_model(cfg.model_name)

        assert self.model is not None
        assert self.model_config is not None

        output_path = render_cache_path(cfg)
        if output_path.exists():
            return output_path

        sample_rate = self.model_config["sample_rate"]
        sample_size = self.model_config["sample_size"]
        total_samples = int(cfg.seconds_total * sample_rate)

        cond_a = get_or_compute_cond(
            self.model, cfg, cfg.prompt_a, self.device, encode_prompt_conditioning
        )
        cond_b = get_or_compute_cond(
            self.model, cfg, cfg.prompt_b, self.device, encode_prompt_conditioning
        )
        cond_c = get_or_compute_cond(
            self.model, cfg, cfg.prompt_c, self.device, encode_prompt_conditioning
        )
        cond_d = get_or_compute_cond(
            self.model, cfg, cfg.prompt_d, self.device, encode_prompt_conditioning
        )

        mixed_cond = mix_anchor_conditioning(
            cond_a,
            cond_b,
            cond_c,
            cond_d,
            x=cfg.x,
            y=cfg.y,
        )

        init_audio = maybe_load_init_audio(cfg, self.device)
        sampler_type = get_sampler_type(cfg.model_name)

        with torch.no_grad():
            output = generate_diffusion_cond(
                self.model,
                steps=cfg.steps,
                cfg_scale=cfg.cfg,
                conditioning_tensors=mixed_cond,
                sample_size=sample_size,
                sample_rate=sample_rate,
                seed=cfg.seed,
                device=self.device,
                sampler_type=sampler_type,
                init_audio=init_audio,
                init_noise_level=cfg.init_noise_level,
            )
        audio = postprocess_output(output, total_samples=total_samples)
        return save_audio(output_path, audio, sample_rate)
