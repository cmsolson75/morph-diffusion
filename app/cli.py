import argparse
from pathlib import Path
from morph.config import Config
from morph.engine import MorphEngine
from morph.presets import PRESETS, get_preset

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Morph cli")

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name or repo id, Defaults ot Config default."
    )
    parser.add_argument("--x", type=float, default=0.5, help="X position in [0, 1]")
    parser.add_argument("--y", type=float, default=0.5, help="Y position in [0, 1]")
    parser.add_argument("--steps", type=int, default=None, help="Sampling steps")
    parser.add_argument("--cfg", type=float, default=None, help="CFG scale")
    parser.add_argument("--seconds", type=float, default=None, help="Output seconds")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=sorted(PRESETS.keys()),
        help="Optional prompt preset"
    )
    parser.add_argument("--prompt-a", type=str, default=None)
    parser.add_argument("--prompt-b", type=str, default=None)
    parser.add_argument("--prompt-c", type=str, default=None)
    parser.add_argument("--prompt-d", type=str, default=None)
    parser.add_argument(
        "--init-audio",
        type=str,
        default=None,
        help="Optional init audio path"
    )
    parser.add_argument(
        "--init-noise-level",
        type=float,
        default=None,
        help="Optional init noise level"
    )

    return parser

def cfg_from_args(args: argparse.Namespace) -> Config:
    cfg = Config()

    if args.model is not None:
        cfg.model_name = args.model

    if args.steps is not None:
        cfg.steps = args.steps
    if args.cfg is not None:
        cfg.cfg = args.cfg
    if args.seconds is not None:
        cfg.seconds_total = args.seconds
    if args.seed is not None:
        cfg.seed = args.seed

    cfg.x = args.x
    cfg.y = args.y

    if args.init_audio is not None:
        cfg.init_audio_path = args.init_audio
    if args.init_noise_level is not None:
        cfg.init_noise_level = args.init_noise_level

    if args.preset is not None:
        preset = get_preset(args.preset)
        cfg.prompt_a = preset["prompt_a"]
        cfg.prompt_b = preset["prompt_b"]
        cfg.prompt_c = preset["prompt_c"]
        cfg.prompt_d = preset["prompt_d"]

    if args.prompt_a is not None:
        cfg.prompt_a = args.prompt_a
    if args.prompt_b is not None:
        cfg.prompt_b = args.prompt_b
    if args.prompt_c is not None:
        cfg.prompt_c = args.prompt_c
    if args.prompt_d is not None:
        cfg.prompt_d = args.prompt_d

    return cfg


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = cfg_from_args(args)
    engine = MorphEngine(model_name=cfg.model_name)
    output_path = engine.render_xy(cfg)

    print(f"output: {Path(output_path).resolve()}")
    print(
        f"model={cfg.model_name} x={cfg.x:.3f} y={cfg.y:.3f} "
        f"steps={cfg.steps} cfg={cfg.cfg} seconds={cfg.seconds_total} seed={cfg.seed}"
    )

if __name__ == "__main__":
    main()