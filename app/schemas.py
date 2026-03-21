from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    model_name: str | None = None
    x: float = Field(default=0.5, ge=0.0, le=1.0)
    y: float = Field(default=0.5, ge=0.0, le=1.0)
    steps: int | None = Field(default=None, ge=1, le=512)
    cfg: float | None = Field(default=None, ge=0.0, le=20.0)
    seconds_total: float | None = Field(default=None, ge=0.1, le=10.0)
    seed: int | None = None

    preset: str | None = None

    prompt_a: str | None = None
    prompt_b: str | None = None
    prompt_c: str | None = None
    prompt_d: str | None = None

    init_audio_path: str | None = None
    init_noise_level: float | None = Field(default=None, ge=0.0, le=10.0)


class GenerateResponse(BaseModel):
    audio_url: str
    output_path: str
    model_name: str
    x: float
    y: float
    steps: int
    cfg: float
    seconds_total: float
    seed: int
