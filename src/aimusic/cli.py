"""Command line entry point.

    aimusic "4 min soothing study music with rain, theme: forest floor"
    aimusic doctor
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import __version__
from .backends import BACKENDS, get_backend
from .nlparse import enrich_with_ollama, parse_prompt
from .schema import Settings
from .soundscape import mix_soundscapes
from .tuning import retune


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aimusic",
        description="Describe music in plain English and generate it.",
        epilog='Other command: "aimusic doctor" checks your environment.',
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
    print(f"  style     : {s.style_prompt}")
    print(f"  duration  : {s.duration:.0f}s ({s.duration / 60:.1f} min)")
    print(f"  vocals    : {'no (instrumental)' if s.lyrics == '[instrumental]' else 'yes'}")
    print(f"  ambience  : {', '.join(s.soundscapes) or 'none'}"
          + (f" @ {s.soundscape_level:.2f}" if s.soundscapes else ""))
    print(f"  tuning    : A = {s.tuning_hz:g} Hz" + (" (shifted after generation)" if s.tuning_hz != 440 else ""))
    print(f"  seed      : {s.seed if s.seed is not None else 'random'}\n")


def _write(audio, sr: int, out: Path) -> None:
    import soundfile as sf

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".mp3":
        if not shutil.which("ffmpeg"):
            raise RuntimeError("MP3 export needs ffmpeg on PATH (run `aimusic doctor`).")
        wav = out.with_suffix(".tmp.wav")
        sf.write(wav, audio, sr)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), str(out)], check=True)
        wav.unlink()
    else:
        sf.write(out, audio, sr)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "doctor":
        from .doctor import run
        return run()

    args = _build_parser().parse_args(argv)
    s = parse_prompt(args.prompt)
    if args.llm:
        s = enrich_with_ollama(s, model=args.llm)
    if args.duration:
        s.duration = args.duration
    if args.seed is not None:
        s.seed = args.seed
    if args.lyrics:
        s.lyrics = args.lyrics.read_text(encoding="utf-8")
    if args.tuning:
        s.tuning_hz = args.tuning
    if args.no_soundscape:
        s.soundscapes = []
    if args.ambience_level is not None:
        s.soundscape_level = args.ambience_level

    _show(s)
    if not args.yes and sys.stdin.isatty():
        if input("Generate? [Y/n] ").strip().lower() in ("n", "no"):
            return 0

    t0 = time.time()
    backend = get_backend(args.backend)
    print(f"Generating with '{backend.name}' backend...")
    audio = backend.generate(s)
    audio = mix_soundscapes(audio, backend.sample_rate, s.soundscapes, s.soundscape_level, s.seed)
    audio = retune(audio, s.tuning_hz)

    slug = re.sub(r"[^a-z0-9]+", "-", args.prompt.lower()).strip("-")[:40] or "track"
    out = args.out or Path("output") / f"{slug}.wav"
    _write(audio, backend.sample_rate, out)
    meta = out.with_suffix(".json")
    meta.write_text(json.dumps({**s.to_dict(), "backend": backend.name,
                                "sample_rate": backend.sample_rate}, indent=2), encoding="utf-8")
    print(f"Done in {time.time() - t0:.1f}s\n  audio   : {out}\n  settings: {meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
