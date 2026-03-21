# MorphDiffusion

MorphDiffusion is a research demo for exploring prompt-space interpolation with Stable Audio Open. The main way to use the repo is to run the local web UI, but the core technical work is in `morph/`: a clean SLERP-style interpolation function over prompt embeddings, plus the conditioning pipeline that injects those mixed embeddings back into Stable Audio generation.

## What this repo is for

This project explores a simple idea: treat four prompts as anchor points in a 2D space, interpolate between their conditioning, and listen to how the generated audio changes as you move around that space.

In practice:

- the primary usage path is the local FastAPI + web UI research demo
- the important implementation work lives in `morph/`
- the UI is there to make the interpolation experiments easy to run and audition

## Run the research demo

### Prerequisites

- Python 3.11+
- `uv`
- a Hugging Face account
- access approval for the Stable Audio model repos you want to use

### Install

```bash
uv sync
```

### Hugging Face setup

You need to do this before the first generation run because the Stable Audio weights are downloaded from gated Hugging Face repos.

1. Sign into Hugging Face.
2. Open these model pages while logged in and accept the Stable Audio access terms / license agreement:
   - https://huggingface.co/stabilityai/stable-audio-open-1.0
   - https://huggingface.co/stabilityai/stable-audio-open-small
3. Create a Hugging Face access token.
4. Authenticate locally:

```bash
uv run hf auth login
```

If you do not complete the Hugging Face approval and Stable Audio signing/access step, model download will fail when the app tries to render audio.

### Start the app

```bash
uv run uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/
```

Useful endpoints:

- `/`: web UI
- `/health`: app status and available presets
- `/presets`: preset list
- `/generate`: generation endpoint
- `/audio/<filename>`: rendered WAV files

## How the demo works

1. The browser UI sends a generation request to the FastAPI app.
2. The app builds a `Config` object from the request.
3. `morph.engine.MorphEngine` loads the selected Stable Audio model.
4. Each anchor prompt is encoded into conditioning tensors.
5. The prompt embeddings are mixed with SLERP-based XY interpolation.
6. The mixed conditioning is passed into Stable Audio diffusion sampling.
7. The rendered WAV is cached and served back through the UI.

## Core research code

The main meat of the repo is in `morph/`:

- `morph/engine.py`: top-level orchestration for loading models, caching, conditioning mix, diffusion generation, and audio save
- `morph/interpolate.py`: clean SLERP-style interpolation utilities
- `morph/conditioning.py`: prompt encoding, model-specific metadata, mask merge, init-audio handling, and conditioning reconstruction
- `morph/models.py`: device selection and sampler/model loading
- `morph/cache.py`: conditioning and render caches
- `morph/config.py`: defaults and model aliases
- `morph/presets.py`: repeatable prompt presets

Supporting layers:

- `app/`: thin FastAPI wrapper over the morph engine
- `web/`: lightweight UI for the research demo
- `notes/`: research notes and references
- `scratch/`: one-off experiments used while developing the approach

## Interpolation design

The most important piece of the repo is the interpolation logic in `morph/interpolate.py`.

The goal was not just to linearly blend prompt tensors, but to build a cleaner SLERP-style function that behaves well on embedding vectors:

- it clamps interpolation values into `[0, 1]`
- it separates direction from magnitude
- it SLERPs the directions
- it linearly interpolates the norms
- it falls back to normalized LERP when vectors are nearly parallel

That gives a practical interpolation with stable endpoint behavior:

- `t = 0` returns approximately the first embedding
- `t = 1` returns approximately the second embedding
- intermediate values move smoothly through embedding space without throwing away magnitude information

The 2D morph is built by composing that same function twice in `slerp_xy(...)`:

- interpolate A → B across `x`
- interpolate C → D across `x`
- interpolate between those two intermediate results across `y`

Conceptually:

```text
A ---- B
|      |
C ---- D
```

So the XY pad is not doing a separate custom blend rule. It is just nested SLERP-style interpolation over four prompt anchors.

## Conditioning design

The second important piece is the conditioning path in `morph/conditioning.py`.

The key design choice is that the repo morphs prompt conditioning before Stable Audio wraps everything into its final diffusion inputs. In other words, the morph happens at the prompt embedding level, not after the full conditioning bundle has already been routed downstream.

Current flow:

1. Build metadata for each prompt.
2. Call `model.conditioner(...)` for each anchor.
3. Extract `cond["prompt"]`, which is an embedding tensor plus its attention mask.
4. Interpolate the prompt embeddings with `slerp_xy(...)`.
5. Merge the prompt masks across anchors.
6. Preserve the rest of the conditioning structure and replace only the prompt portion.
7. Pass the reconstructed conditioning tensors into diffusion sampling.

Why this matters:

- the prompt embedding is the morphable latent representation
- the non-prompt conditioning should stay structurally valid
- mixing earlier in the pipeline is cleaner than trying to morph wrapper-level tensors after they have already been assembled

There is also model-specific metadata handling here:

- `stable-audio-open-small` uses prompt + `seconds_total`
- `stable-audio-open-1.0` uses prompt + `seconds_start` + `seconds_total`

`init_audio_path` is also handled in this layer. If provided, the app loads local audio from the server machine and uses it as init audio for diffusion.

## API example

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "stabilityai/stable-audio-open-1.0",
    "x": 0.5,
    "y": 0.5,
    "steps": 36,
    "cfg": 5.0,
    "seconds_total": 3.0,
    "seed": 1234,
    "prompt_a": "short synthetic bass hit, distorted midrange, punchy transient, tight tail, sound design one-shot",
    "prompt_b": "short digital glitch burst, noisy high frequency texture, sharp transient, granular tail, synthetic one-shot",
    "prompt_c": "short metallic percussion hit, resonant body, crisp attack, tight decay, synthetic one-shot",
    "prompt_d": "short airy noise stab, bright textured transient, spectral tail, abstract synthetic one-shot"
  }'
```

Response fields include:

- `audio_url`
- `output_path`
- `model_name`
- `x`
- `y`
- `steps`
- `cfg`
- `seconds_total`
- `seed`

## Secondary path: CLI

The CLI is useful for direct experiments, but it is secondary to the web research demo:

```bash
uv run python -m app.cli --help
```

## Notes

- device selection is automatic: MPS, then CUDA, then CPU
- models are loaded lazily on first generation
- first-run latency is higher because weights may need to download
- `.cache/cond/` stores prompt conditioning cache entries
- `.cache/render/` stores rendered WAV outputs
- `init_audio_path` must point to a file on the same machine that is running the FastAPI app
