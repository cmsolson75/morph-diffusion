from pathlib import Path

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import FileResponse

from morph.config import Config
from morph.engine import MorphEngine
from morph.presets import PRESETS, get_preset


from .schemas import GenerateRequest, GenerateResponse

router = APIRouter()


def cfg_from_request(req: GenerateRequest) -> Config:
    cfg = Config()

    if req.model_name is not None:
        cfg.model_name = req.model_name
    if req.steps is not None:
        cfg.steps = req.steps
    if req.cfg is not None:
        cfg.cfg = req.cfg
    if req.seconds_total is not None:
        cfg.seconds_total = req.seconds_total
    if req.seed is not None:
        cfg.seed = req.seed

    cfg.x = req.x
    cfg.y = req.y

    if req.init_audio_path is not None:
        cfg.init_audio_path = req.init_audio_path
    if req.init_noise_level is not None:
        cfg.init_noise_level = req.init_noise_level

    if req.preset is not None:
        if req.preset not in PRESETS:
            raise HTTPException(status_code=400, detail=f"Unknown preset: {req.preset}")
        preset = get_preset(req.preset)
        cfg.prompt_a = preset["prompt_a"]
        cfg.prompt_b = preset["prompt_b"]
        cfg.prompt_c = preset["prompt_c"]
        cfg.prompt_d = preset["prompt_d"]

    if req.prompt_a is not None:
        cfg.prompt_a = req.prompt_a
    if req.prompt_b is not None:
        cfg.prompt_b = req.prompt_b
    if req.prompt_c is not None:
        cfg.prompt_c = req.prompt_c
    if req.prompt_d is not None:
        cfg.prompt_d = req.prompt_d

    return cfg


@router.get("/health")
async def health(request: Request) -> dict:
    engine: MorphEngine = request.app.state.engine
    return {
        "status": "ok",
        "device": engine.device,
        "loaded_model": engine._loaded_name,
        "available_presets": sorted(PRESETS.keys()),
    }


@router.get("/presets")
async def presets() -> dict:
    return {"presets": sorted(PRESETS.keys())}


@router.post("/generate", response_model=GenerateResponse)
async def generate(request: Request, body: GenerateRequest) -> GenerateResponse:
    engine: MorphEngine = request.app.state.engine
    cfg = cfg_from_request(body)

    output_path = engine.render_xy(cfg)
    rel = output_path.as_posix()
    return GenerateResponse(
        audio_url=f"/audio/{output_path.name}",
        output_path=rel,
        model_name=cfg.model_name,
        x=cfg.x,
        y=cfg.y,
        steps=cfg.steps,
        cfg=cfg.cfg,
        seconds_total=cfg.seconds_total,
        seed=cfg.seed,
    )


@router.get("/audio/{filename}")
async def get_audio(filename: str) -> FileResponse:
    path = Path(".cache/render") / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(path, media_type="audio/wav")
