from stable_audio_tools import get_pretrained_model
import argparse

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="stabilityai/stable-audio-open-1.0")
    return parser

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    print(f"Loading model: {args.model}")
    model = get_pretrained_model(args.model)
    print("Model loaded successfully")

if __name__ == "__main__":
    main()