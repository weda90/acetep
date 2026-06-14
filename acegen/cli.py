import argparse
import sys

from .generate import generate


def build_parser():
    p = argparse.ArgumentParser(
        prog="acegen",
        description="Generate music with ACE-Step1.5 MLX",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # generate
    g = sub.add_parser("generate", help="Generate music from text prompt")
    g.add_argument("-t", "--text", required=True, help="Music description prompt")
    g.add_argument("-l", "--lyrics", default="", help="Lyrics text or .txt file path")
    g.add_argument("-d", "--duration", type=float, default=30.0, help="Duration in seconds (default: 30)")
    g.add_argument("-o", "--output", default="output.wav", help="Output WAV file (default: output.wav)")
    g.add_argument("-L", "--lang", default="en", help="Vocal language code (en, id, zh, etc.)")
    g.add_argument("--no-lm", action="store_false", dest="use_lm", help="Disable 5Hz LM")
    g.add_argument("--lm-model", default="0.6B", choices=["0.6B", "4B"], help="5Hz LM size (default: 0.6B)")
    g.add_argument("--steps", type=int, default=8, help="Diffusion steps (default: 8)")
    g.add_argument("--shift", type=float, default=3.0, help="Timestep shift (default: 3.0)")
    g.add_argument("--cfg", type=float, default=1.0, help="Guidance scale (default: 1.0)")
    g.add_argument("--bpm", type=int, default=None, help="BPM override")
    g.add_argument("--key", default=None, help="Key/scale override")
    g.add_argument("--seed", type=int, default=None, help="Random seed")
    g.add_argument("--model", default="mlx-community/ACE-Step1.5-MLX-4bit", help="HF model ID")
    g.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "generate":
        generate(
            text=args.text,
            lyrics=args.lyrics,
            duration=args.duration,
            output=args.output,
            vocal_language=args.lang,
            use_lm=args.use_lm,
            lm_model_size=args.lm_model,
            num_steps=args.steps,
            shift=args.shift,
            guidance_scale=args.cfg,
            bpm=args.bpm,
            keyscale=args.key,
            seed=args.seed,
            model_id=args.model,
            verbose=args.verbose,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
