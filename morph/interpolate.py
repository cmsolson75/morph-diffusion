from __future__ import annotations
import torch


def normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / torch.linalg.norm(x, dim=-1, keepdim=True).clamp(min=eps)


def clamp01(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


def slerp(
    a: torch.Tensor, b: torch.Tensor, t: float, eps: float = 1e-6
) -> torch.Tensor:
    """
    Spherical interpolation that preserves endpoint magnitudes by:
    1. SLERPing directions
    2. Linearly interpolating norms

    This gives:
    - t = 0.0 -> approximately a
    - t = 1.0 -> approximately b
    """
    t = clamp01(t)

    a_norm = torch.linalg.norm(a, dim=-1, keepdim=True).clamp(min=eps)
    b_norm = torch.linalg.norm(b, dim=-1, keepdim=True).clamp(min=eps)

    a_dir = a / a_norm
    b_dir = b / b_norm

    dot = (a_dir * b_dir).sum(dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)

    # Fallback for nearly parallel vectors
    dir_lerp = normalize((1.0 - t) * a_dir + t * b_dir, eps=eps)

    dir_slerp = (
        torch.sin((1.0 - t) * omega) / sin_omega * a_dir
        + torch.sin(t * omega) / sin_omega * b_dir
    )

    near = sin_omega.abs() < 1e-4
    mixed_dir = torch.where(near, dir_lerp, dir_slerp)

    mixed_norm = (1.0 - t) * a_norm + t * b_norm
    return mixed_dir * mixed_norm


def slerp_xy(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    x: float,
    y: float,
) -> torch.Tensor:
    x = clamp01(x)
    y = clamp01(y)

    bottom = slerp(a, b, x)
    top = slerp(c, d, x)
    return slerp(bottom, top, y)
