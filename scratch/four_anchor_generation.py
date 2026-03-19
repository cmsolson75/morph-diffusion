from dataclasses import dataclass

import torch
import torchaudio
from einops import rearrange

from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
import argparse

MODELS = {
    "small": "stabilityai/stable-audio-open-small",
    "main": "stabilityai/stable-audio-open-1.0",
}


@dataclass
class Cfg:
    model_name: str = MODELS["main"]
    steps: int = 36
    cfg: float = 5.0
    seconds_total: float = 3.0
    x: float = 0.1
    y: float = 0.0
    seed: int = 1234

    # Corner layout:
    # A ---- B
    # |      |
    # C ---- D
    #
    # A = (0,0), B = (1,0), C = (0,1), D = (1,1)

    prompt_a: str = (
        "short synthetic bass hit, distorted midrange, punchy transient, "
        "tight tail, sound design one-shot"
    )

    prompt_b: str = (
        "short digital glitch burst, noisy high frequency texture, sharp transient, "
        "granular tail, synthetic one-shot"
    )

    prompt_c: str = (
        "short metallic percussion hit, resonant body, crisp attack, "
        "tight decay, synthetic one-shot"
    )

    prompt_d: str = (
        "short airy noise stab, bright textured transient, spectral tail, "
        "abstract synthetic one-shot"
    )


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


def build_metadata(cfg: Cfg, prompt: str) -> list[dict]:
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-x", type=float)
    parser.add_argument("-y", type=float)
    return parser.parse_args()


def main() -> None:
    cfg = Cfg()
    args = parse_args()
    cfg.x = args.x
    cfg.y = args.y

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    model, model_config = get_pretrained_model(cfg.model_name)
    model = model.to(device)

    if device in {"mps", "cpu"}:
        model.pretransform.model_half = False
        model = model.to(torch.float32)

    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    total_samples = int(cfg.seconds_total * sample_rate)

    meta_a = build_metadata(cfg, cfg.prompt_a)
    meta_b = build_metadata(cfg, cfg.prompt_b)
    meta_c = build_metadata(cfg, cfg.prompt_c)
    meta_d = build_metadata(cfg, cfg.prompt_d)

    cond_a = model.conditioner(meta_a, device=device)
    cond_b = model.conditioner(meta_b, device=device)
    cond_c = model.conditioner(meta_c, device=device)
    cond_d = model.conditioner(meta_d, device=device)

    prompt_a, mask_a = cond_a["prompt"]
    prompt_b, mask_b = cond_b["prompt"]
    prompt_c, mask_c = cond_c["prompt"]
    prompt_d, mask_d = cond_d["prompt"]

    mixed_prompt = slerp_xy(prompt_a, prompt_b, prompt_c, prompt_d, x=cfg.x, y=cfg.y)
    mixed_mask = mask_a | mask_b | mask_c | mask_d

    # Preserve all non-prompt conditioning exactly as produced by cond_a
    mixed_cond = dict(cond_a)
    mixed_cond["prompt"] = (mixed_prompt, mixed_mask)

    sampler_type = "dpmpp-2m-sde" if cfg.model_name == MODELS["main"] else "pingpong"

    with torch.no_grad():
        output = generate_diffusion_cond(
            model,
            steps=cfg.steps,
            cfg_scale=cfg.cfg,
            conditioning_tensors=mixed_cond,
            sample_size=sample_size,
            sample_rate=sample_rate,
            seed=cfg.seed,
            device=device,
            sampler_type=sampler_type,
        )

    output = rearrange(output, "b d n -> d (b n)")
    output = output.to(torch.float32)
    output = output[:, :total_samples]
    output = output / output.abs().max().clamp(min=1e-8)

    torchaudio.save("morph_output_xy.wav", output.cpu(), sample_rate)
    print("Saved morph_output_xy.wav")


if __name__ == "__main__":
    main()
