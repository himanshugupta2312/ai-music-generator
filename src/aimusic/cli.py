"""Command line entry point.

    aimusic "4 min soothing study music with rain, theme: forest floor"
    aimusic ui          # open the web UI (needs: pip install "aimusic[ui]")
    aimusic doctor
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

from . import __version__
from .backends import BACKENDS, get_backend
from .pipeline import MAX_DURATION_S, build_settings, describe, render, save, validate  # noqa: F401
from .schema import Settings


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aimusic",
        description="Describe music in plain English and generate it.",
        epilog='Other commands: "aimusic ui" opens the web UI, "aimusic doctor" checks your environment.',
    )
    p.add_argument("prompt", help="plain-English description of the music")
    p.add_argument("--backend", choices=BACKENDS, default="synth")
    p.add_argument("--duration", type=float, help="seconds (overrides the prompt)")
    p.add_argument("--seed", type=int, help="reproducible results")
    p.add_argument("--lyrics", type=Path, help="text file with lyrics ([verse]/[chorus] tags)")
    p.add_argument("--tuning", type=float, metavar="HZ", help="concert A in Hz (default 440)")
    p.add_argument("--no-soundscape", action="store_true", help="skip ambience layers")
    p.add_argument("--ambience-level", type=float, help="0..1 ambience volume (default 0.35)")
    p.add_argument("--llm", metavar="OLLAMA_MODEL", help="enrich style prompt with a local Ollama model")
    p.add_argument("--out", type=Path, help="output file (.wav or .mp3)")
    p.add_argument("-y", "--yes", action="store_true", help="don't ask for confirmation")
    p.add_argument("--version", action="version", version=f"aimusic {__version__}")
    return p


def _show(s: Settings) -> None:
    print("\nInterpreted request:")
    for label, value in describe(s):
        print(f"  {label:<10}: {value}")
    print()


def _run_ui(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="aimusic ui", description="Open the aimusic web UI (local only).")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args(argv)
    try:
        from .ui import launch
    except ImportError:
        print('The web UI needs Gradio. Install it with:  pip install "aimusic[ui]"', file=sys.stderr)
        return 1
    launch(port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "doctor":
        from .doctor import run
        return run()
    if argv and argv[0] == "ui":
        return _run_ui(argv[1:])

    args = _build_parser().parse_args(argv)
    s = build_settings(
        args.prompt,
        duration=args.duration,
        seed=args.seed,
        lyrics=args.lyrics.read_text(encoding="utf-8") if args.lyrics else None,
        tuning=args.tuning,
        ambience=not args.no_soundscape,
        ambience_level=args.ambience_level,
        llm=args.llm,
    )
    try:
        warnings = validate(s, args.backend)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    for w in warnings:
        print(f"[note] {w}")

    _show(s)
    if not args.yes and sys.stdin.isatty():
        if input("Generate? [Y/n] ").strip().lower() in ("n", "no"):
            return 0

    t0 = time.time()
    backend = get_backend(args.backend)
    print(f"Generating with '{backend.name}' backend...")
    audio = render(backend, s)

    slug = re.sub(r"[^a-z0-9]+", "-", args.prompt.lower()).strip("-")[:40] or "track"
    out = args.out or Path("output") / f"{slug}.wav"
    meta = save(audio, backend.sample_rate, out, s, backend.name)
    print(f"Done in {time.time() - t0:.1f}s\n  audio   : {out}\n  settings: {meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
