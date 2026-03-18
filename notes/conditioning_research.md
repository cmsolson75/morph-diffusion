# MorphDiffusion

## Model / Environment
- Model: stabilityai/stable-audio-open-small
- Library: stable-audio-tools
- Device: MPS (Mac)
- Generation via generate_diffusion_cond(...) works

## Conditioning Pipeline
conditioning happens in two stages:
metadata → model.conditioner(...) → model.get_conditioning_inputs(...) → diffusion model

## Conditioner Output
cond["prompt"]:
  embed: (1, 64, 768) float32
  mask:  (1, 64) bool

cond["seconds_total"]:
  embed: (1, 1, 768)
  mask:  (1, 1)

## Wrapper Output (model.get_conditioning_inputs)
cross_attn_cond: (1, 65, 768)
cross_attn_mask: (1, 65)
global_cond:     (1, 768)

input_concat_cond: None
prepend_cond: None

## Routing
cross_attn_cond_ids = ['prompt', 'seconds_total']
global_cond_ids     = ['seconds_total']

→ prompt and seconds_total are concatenated into cross-attention
→ seconds_total also becomes global conditioning

## Key Insight
- Prompt embedding (1, 64, 768) is the morphable latent space
- Wrapper appends seconds_total as an extra token → total 65 tokens
- Must morph BEFORE wrapper stage

## Morph Strategy
1. Encode anchors:
   cond_a = model.conditioner(meta_a)
   cond_b = model.conditioner(meta_b)

2. Extract:
   prompt_a, mask_a = cond_a["prompt"]
   prompt_b, mask_b = cond_b["prompt"]

3. Interpolate:
   mixed_prompt = slerp(prompt_a, prompt_b, t)

4. Rebuild conditioning:
   mixed_cond = {
     "prompt": (mixed_prompt, mask_a),
     "seconds_total": cond_a["seconds_total"]
   }

5. Wrap:
   wrapped = model.get_conditioning_inputs(mixed_cond)

## Goal
Inject `wrapped` directly into diffusion sampling
→ bypass text prompt pathway entirely

## Open Problem
generate_diffusion_cond(...) internally rebuilds conditioning
Need to:
- either pass custom conditioning into sampler
- or patch generation path

## Notes
- Output length is fixed (~11s); seconds_total only biases placement
- Trim waveform manually to desired duration
- High steps (>50) destabilize output
- Quality is not critical yet; correctness of conditioning is priority
- Attn Masks are not equal length - this will require some work.