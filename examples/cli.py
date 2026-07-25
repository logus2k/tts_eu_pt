#!/usr/bin/env python3
"""Command-line synthesis: text -> WAV.

    python -m examples.cli "Olá! São dezasseis horas." ola.wav
    python -m examples.cli --speed 1.1 --text "Bom dia." out.wav
"""
import argparse
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="European Portuguese TTS (tts_eu_pt).")
    ap.add_argument("text", nargs="?", help="text to speak (or use --text)")
    ap.add_argument("out", nargs="?", default="out.wav", help="output WAV path")
    ap.add_argument("--text", dest="text_opt", help="text to speak")
    ap.add_argument("--speed", type=float, default=1.0, help=">1 faster, <1 slower")
    ap.add_argument("--device", default=None, help="cuda | cpu (auto if omitted)")
    ap.add_argument("--model", default=None, help="local model .pth (skips download)")
    ap.add_argument("--voicepack", default=None, help="local voicepack .pt (skips download)")
    args = ap.parse_args(argv)

    text = args.text_opt or args.text
    if not text:
        ap.error("provide text as a positional arg or via --text")

    from tts_eu_pt import TTS

    tts = TTS(device=args.device, model_path=args.model, voicepack_path=args.voicepack)
    path = tts.save(args.out, text, speed=args.speed)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
