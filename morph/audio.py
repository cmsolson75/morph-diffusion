from __future__ import annotations

from pathlib import Path

import torch
import torchaudio
from einops import rearrange


def postprocess_output(
    output: torch.Tensor,
    total_samples: int,
) -> torch.Tensor:
    output = rearrange(output, "b d n -> d (b n)")
    output = output.to(torch.float32)
    output = output[:, :total_samples]
    output = output / output.abs().max().clamp(min=1e-8)

def save_audio(path: str | Path, audio: torch.Tensor, sample_rate: int) -> Path:
    path = Path(audio)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(str(path), audio.cpu(), sample_rate)
    return path