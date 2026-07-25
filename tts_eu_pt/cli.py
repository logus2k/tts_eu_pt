#!/usr/bin/env python3
"""Command-line synthesis: text -> WAV.

    tts-eu-pt "Olá! São dezasseis horas." ola.wav
    tts-eu-pt --text "Bom dia." --speed 1.1 ola.wav
    python -m tts_eu_pt.cli "Bom dia." ola.wav

This lives inside the package (not in examples/) so the ``tts-eu-pt`` console script
declared in pyproject.toml resolves after a plain ``pip install tts_eu_pt``.
"""
import argparse
import sys

DEFAULT_OUT = "out.wav"


def _parse(argv):
    """Resolve (text, out) from the command line.

    Text may be given positionally or with --text. The positionals are collected as a
    plain list rather than as two optional positionals, because argparse binds a lone
    positional to the FIRST optional slot: with ``text``/``out`` both nargs="?", the
    command ``--text "Bom dia." ola.wav`` put "ola.wav" into `text` (where it was then
    discarded in favour of --text) and silently wrote to the default out.wav instead.
    """
    ap = argparse.ArgumentParser(
        prog="tts-eu-pt",
        description="European Portuguese TTS (tts_eu_pt).",
        epilog='examples: tts-eu-pt "Bom dia." ola.wav  |  tts-eu-pt --text "Bom dia." ola.wav',
    )
    ap.add_argument("words", nargs="*", metavar="TEXT [OUT]",
                    help=f"text to speak, then the output WAV path (default {DEFAULT_OUT})")
    ap.add_argument("--text", dest="text_opt", help="text to speak (alternative to positional)")
    ap.add_argument("--out", dest="out_opt", help=f"output WAV path (default {DEFAULT_OUT})")
    ap.add_argument("--speed", type=float, default=1.0, help=">1 faster, <1 slower")
    ap.add_argument("--device", default=None, help="cuda | cpu (auto if omitted)")
    ap.add_argument("--model", default=None, help="local model .pth (skips download)")
    ap.add_argument("--voicepack", default=None, help="local voicepack .pt (skips download)")
    args = ap.parse_args(argv)

    words = list(args.words)
    if args.text_opt is not None:
        # --text supplies the text, so every positional is the output path.
        text = args.text_opt
        if len(words) > 1:
            ap.error(f"expected at most one output path after --text, got {len(words)}: {words}")
        out = words[0] if words else None
    else:
        if not words:
            ap.error("provide text as a positional arg or via --text")
        text = words[0]
        if len(words) > 2:
            ap.error(f"expected at most TEXT and OUT, got {len(words)}: {words}")
        out = words[1] if len(words) > 1 else None

    if args.out_opt is not None:
        if out is not None:
            ap.error(f"output path given twice: {out!r} and --out {args.out_opt!r}")
        out = args.out_opt

    if not text.strip():
        ap.error("refusing to synthesise empty text")

    return args, text, out or DEFAULT_OUT


def main(argv=None) -> int:
    args, text, out = _parse(argv)

    from tts_eu_pt import TTS

    tts = TTS(device=args.device, model_path=args.model, voicepack_path=args.voicepack)
    path = tts.save(out, text, speed=args.speed)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
