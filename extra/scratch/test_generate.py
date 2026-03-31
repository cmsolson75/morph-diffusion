import torch
import torchaudio
from einops import rearrange

from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

MODEL_NAME = "stabilityai/stable-audio-open-small"


def main():
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")
    model, model_config = get_pretrained_model(MODEL_NAME)
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]

    seconds_total = 3.0
    total_samples = int(seconds_total * sample_rate)

    model = model.to(device)

    if device in {"mps", "cpu"}:
        model.pretransform.model_half = False
        model = model.to(torch.float32)

    conditioning = [
        {
            "prompt": "short metallic impact, sharp transient, tight decay, dry, high frequency detail, designed percussion one-shot",
            "seconds_total": seconds_total,
        }
    ]

    with torch.no_grad():
        output = generate_diffusion_cond(
            model,
            steps=24,
            cfg_scale=3.0,
            conditioning=conditioning,
            sample_size=sample_size,
            sampler_type="pingpong",
            device=device,
        )
    output = rearrange(output, "b d n -> d (b n)")
    output = output.to(torch.float32)
    output = output[:, :total_samples]
    output = output / output.abs().max().clamp(min=1e-8)

    torchaudio.save("output.wav", output.cpu(), sample_rate)
    print("Saved output.wav")


if __name__ == "__main__":
    main()
