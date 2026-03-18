from dataclasses import dataclass

import torch
import torchaudio
from einops import rearrange

from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

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
    t: float = 0.5
    seed: int = 1234
    prompt_a: str = (
            "short synthetic bass hit, distorted midrange, punchy transient, "
            "tight tail, sound design one-shot"
        )

    prompt_b: str = (
        "short digital glitch burst, noisy high frequency texture, sharp transient, "
        "granular tail, synthetic one-shot"
    )


def normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / torch.linalg.norm(x, dim=-1, keepdim=True).clamp(min=eps)


def slerp(a: torch.Tensor, b: torch.Tensor, t: float, eps: float = 1e-6) -> torch.Tensor:
    """
    Spherical interpolation that preserves endpoint magnitudes by:
    1. SLERPing directions
    2. Linearly interpolating norms

    This gives:
    - t = 0.0 -> approximately a
    - t = 1.0 -> approximately b
    """
    t = float(max(0.0, min(1.0, t)))

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


def main() -> None:
    cfg = Cfg()

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

    cond_a = model.conditioner(meta_a, device=device)
    cond_b = model.conditioner(meta_b, device=device)

    prompt_a, mask_a = cond_a["prompt"]
    prompt_b, mask_b = cond_b["prompt"]

    mixed_prompt = slerp(prompt_a, prompt_b, cfg.t)
    mixed_mask = torch.logical_or(mask_a, mask_b)

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

    torchaudio.save("morph_output.wav", output.cpu(), sample_rate)
    print("Saved morph_output.wav")

    # Optional debug sanity checks
    test0 = slerp(prompt_a, prompt_b, 0.0)
    test1 = slerp(prompt_a, prompt_b, 1.0)
    print("t=0 vs a:", (test0 - prompt_a).abs().mean().item())
    print("t=1 vs b:", (test1 - prompt_b).abs().mean().item())


if __name__ == "__main__":
    main()