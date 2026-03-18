import torch
import torch.nn.functional as F

from stable_audio_tools import get_pretrained_model


MODEL_NAME = "stabilityai/stable-audio-open-small"


def normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    norm = torch.linalg.norm(x, dim=-1, keepdim=True).clamp(min=eps)
    return x / norm


def slerp(
    a: torch.Tensor, b: torch.Tensor, t: float, eps: float = 1e-6
) -> torch.Tensor:
    a_n = normalize(a)
    b_n = normalize(b)

    dot = (a_n * b_n).sum(dim=-1, keepdim=True).clamp(-1 + eps, 1 - eps)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)

    lerp = normalize((1.0 - t) * a + t * b)

    slerp_out = (
        torch.sin((1.0 - t) * omega) / sin_omega * a_n
        + torch.sin(t * omega) / sin_omega * b_n
    )

    near = sin_omega.abs() < 1e-4
    return torch.where(near, lerp, slerp_out)


def main():
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, _ = get_pretrained_model(MODEL_NAME)
    model = model.to(device)

    if device in {"mps", "cpu"}:
        model.pretransform.model_half = False
        model = model.to(torch.float32)

    meta_a = [
        {
            "prompt": "short metallic impact, sharp transient, dry",
            "seconds_total": 2.0,
        }
    ]
    meta_b = [
        {
            "prompt": "short digital glitch burst, noisy high frequency, synthetic",
            "seconds_total": 2.0,
        }
    ]

    cond_a = model.conditioner(meta_a, device=device)
    cond_b = model.conditioner(meta_b, device=device)

    prompt_a, mask_a = cond_a["prompt"]
    prompt_b, mask_b = cond_b["prompt"]

    print("prompt_a:", prompt_a.shape)
    print("prompt_b:", prompt_b.shape)

    print("mask equal:", torch.equal(mask_a, mask_b))

    t = 0.5
    mixed_prompt = slerp(prompt_a, prompt_b, t)
    mixed_mask = mask_a
    mixed_seconds = cond_a["seconds_total"]

    mixed_cond = {
        "prompt": (mixed_prompt, mixed_mask),
        "seconds_total": mixed_seconds,
    }

    wrapped = model.get_conditioning_inputs(mixed_cond)

    print("wrapped keys:", wrapped.keys())
    print("cross_attn_cond:", wrapped["cross_attn_cond"].shape)
    print("cross_attn_mask:", wrapped["cross_attn_mask"].shape)
    print("global_cond:", wrapped["global_cond"].shape)


if __name__ == "__main__":
    main()
