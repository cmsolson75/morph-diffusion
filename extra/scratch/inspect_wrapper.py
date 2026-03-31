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
    model = model.to(device)

    if device in {"mps", "cpu"}:
        model.pretransform.model_half = False
        model = model.to(torch.float32)

    metadata = [
        {
            "prompt": "short metallic impact, sharp transient, dry",
            "seconds_total": 2.0,
        }
    ]

    cond = model.conditioner(metadata, device=device)
    print("cross_attn_cond_ids:", model.cross_attn_cond_ids)
    print("global_cond_ids:", model.global_cond_ids)
    print("input_concat_ids:", model.input_concat_ids)
    print("prepend_cond_ids:", model.prepend_cond_ids)

    print("\nConditioner output:")
    describe(cond, "cond")

    print("\nWrapper conditioning inputs:")
    wrapped = model.get_conditioning_inputs(cond)
    describe(wrapped, "wrapped")

    # print(type(model))
    # for a in dir(model):
    #     if "cond" in a.lower() or "input" in a.lower():
    #         print(a)


if __name__ == "__main__":
    main()
