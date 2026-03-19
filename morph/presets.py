# MOVE TO JSON
PRESETS = {
    "synthetic_bass": {
        "prompt_a": (
            "short synthetic bass hit, distorted midrange, punchy transient, "
            "tight tail, sound design one-shot"
        ),
        "prompt_b": (
            "short digital glitch burst, noisy high frequency texture, sharp transient, "
            "granular tail, synthetic one-shot"
        ),
        "prompt_c": (
            "short metallic percussion hit, resonant body, crisp attack, "
            "tight decay, synthetic one-shot"
        ),
        "prompt_d": (
            "short airy noise stab, bright textured transient, spectral tail, "
            "abstract synthetic one-shot"
        ),
    },
    "industrial_glitch": {
        "prompt_a": (
            "short distorted metal impact, harsh midrange, sharp transient, "
            "tight tail, synthetic one-shot"
        ),
        "prompt_b": (
            "short electrical crackle burst, noisy bright texture, clipped transient, "
            "glitchy tail, synthetic one-shot"
        ),
        "prompt_c": (
            "short mechanical hit, resonant body, punchy attack, "
            "tight decay, sound design one-shot"
        ),
        "prompt_d": (
            "short noisy spectral burst, airy texture, diffuse tail, "
            "abstract synthetic one-shot"
        ),
    },
}


def get_preset(name: str) -> dict[str, str]:
    if name not in PRESETS:
        raise KeyError(f"Unknown preset: {name}")
    return PRESETS[name]
