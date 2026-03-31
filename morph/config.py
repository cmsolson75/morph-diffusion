from dataclasses import dataclass


@dataclass(frozen=True)
class ModelOption:
    id: str
    label: str


AVAILABLE_MODELS: tuple[ModelOption, ...] = (
    ModelOption(
        id="stabilityai/stable-audio-open-1.0",
        label="Stable Audio Open 1.0",
    ),
)

DEFAULT_MODEL_NAME = AVAILABLE_MODELS[0].id


def get_available_models() -> list[dict[str, str]]:
    return [{"id": model.id, "label": model.label} for model in AVAILABLE_MODELS]


def is_supported_model(model_name: str) -> bool:
    return any(model.id == model_name for model in AVAILABLE_MODELS)


@dataclass
class Config:
    model_name: str = DEFAULT_MODEL_NAME
    steps: int = 36
    cfg: float = 5.0
    seconds_total: float = 3.0
    x: float = 0.5
    y: float = 0.5
    seed: int = 1234

    init_audio_path: str | None = None
    init_noise_level: float = 0.0  # lower = preserve more of input audio

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
