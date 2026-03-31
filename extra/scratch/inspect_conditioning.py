import torch
from stable_audio_tools import get_pretrained_model


MODEL_NAME = "stabilityai/stable-audio-open-small"


def describe(obj, prefix="root"):
    if torch.is_tensor(obj):
        print(
            f"{prefix}: tensor shape={tuple(obj.shape)} dtype={obj.dtype} device={obj.device}"
        )

    elif isinstance(obj, dict):
        print(f"{prefix}: dict keys={list(obj.keys())}")
        for k, v in obj.items():
            describe(v, f"{prefix}.{k}")
    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}: {type(obj).__name__} len={len(obj)}")
        for i, v in enumerate(obj):
            describe(v, f"{prefix}[{i}]")
    else:
        print(f"{prefix}: {type(obj).__name__} -> {obj}")


def main():
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")
    model, _ = get_pretrained_model(MODEL_NAME)

    seconds_total = 3.0
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

    print(f"Model type: {type(model)}")
    print(f"Conditioning-related attrs:")
    for a in dir(model):
        if "cond" in a.lower():
            print("  ", a)

    for attr in ["conditioner", "conditioning", "get_conditioning_inputs"]:
        if hasattr(model, attr):
            fn = getattr(model, attr)
            print(f"\nTrying model.{attr} ...")
            try:
                out = fn(conditioning, device=device)
            except TypeError:
                out = fn(conditioning)
            describe(out, prefix=f"model.{attr}()")
            return
    print("\nNo direct conditioning entrypoint fount.")


if __name__ == "__main__":
    main()
