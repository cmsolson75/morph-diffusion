import inspect

from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.inference import sampling

print("generate_diffusion_cond signature:")
print(inspect.signature(generate_diffusion_cond))
print()

print("sample signature")
print(inspect.signature(sampling.sample))
print()

if hasattr(sampling, "sample_k"):
    print("sample_k signature:")
    print(inspect.signature(sampling.sample_k))
    print()

if hasattr(sampling, "sample_rf"):
    print("sample_rf signature:")
    print(inspect.signature(sampling.sample_rf))
    print()

print(
    "generate_diffusion_cond source file:",
    inspect.getsourcefile(generate_diffusion_cond),
)
print("sample source file:", inspect.getsourcefile(sampling.sample))
