"""Shared generation steps used by both the command line and the web UI."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .nlparse import enrich_with_ollama, parse_prompt
from .schema import Settings
from .soundscape import mix_soundscapes
from .tuning import retune

MAX_DURATION_S = 600  # 10-minute cap (ACE-Step's documented per-generation maximum)
MIN_DURATION_S = 5


def build_settings(
    prompt: str,
    *,
    duration: float | None = None,
    seed: int | None = None,
    lyrics: str | None = None,
    tuning: float | None = None,
    ambience: bool = True,
    ambience_level: float | None = None,
    llm: str | None = None,
) -> Settings:
    """Parse a prompt and apply explicit overrides. Falsy overrides mean 'keep the prompt's value'."""
    s = parse_prompt(prompt)
    if llm:
        s = enrich_with_ollama(s, model=llm)
    if duration:
        s.duration = float(duration)
    if seed is not None:
        s.seed = int(seed)
    if lyrics:
        s.lyrics = lyrics
    if tuning:
        s.tuning_hz = float(tuning)
    if not ambience:
        s.soundscapes = []
    if ambience_level is not None:
        s.soundscape_level = float(ambience_level)
    return s


def validate(s: Settings, backend_name: str) -> list[str]:
    """Raise ValueError for unusable settings; return a list of non-fatal warnings."""
    if s.duration > MAX_DURATION_S:
        raise ValueError(
            f"Duration {s.duration:.0f}s is over the {MAX_DURATION_S // 60}-minute limit."
        )
    if s.duration < MIN_DURATION_S:
        raise ValueError(f"Duration must be at least {MIN_DURATION_S} seconds.")
    warnings = []
    if backend_name == "musicgen" and s.duration > 30:
        warnings.append(
            "MusicGen builds tracks over 30 s from separate chunks, so they will not be "
            "continuous. Use the acestep backend for one continuous piece."
        )
    return warnings


def describe(s: Settings) -> list[tuple[str, str]]:
    """Human-readable (label, value) pairs showing how the prompt was understood."""
    return [
        ("style", s.style_prompt),
        ("duration", f"{s.duration:.0f}s ({s.duration / 60:.1f} min)"),
        ("vocals", "no (instrumental)" if s.lyrics == "[instrumental]" else "yes"),
        ("ambience", (", ".join(s.soundscapes) + f" @ {s.soundscape_level:.2f}") if s.soundscapes else "none"),
        ("tuning", f"A = {s.tuning_hz:g} Hz" + (" (shifted after generation)" if s.tuning_hz != 440 else "")),
        ("seed", str(s.seed) if s.seed is not None else "random"),
    ]


def render(backend, s: Settings) -> np.ndarray:
    """Generate audio, then mix in ambience and apply tuning. Returns (samples, channels)."""
    audio = backend.generate(s)
    audio = mix_soundscapes(audio, backend.sample_rate, s.soundscapes, s.soundscape_level, s.seed)
    return retune(audio, s.tuning_hz)


def save(audio: np.ndarray, sr: int, out: Path, s: Settings, backend_name: str) -> Path:
    """Write the audio (.wav or .mp3) and a settings .json next to it. Returns the .json path."""
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
    meta = out.with_suffix(".json")
    meta.write_text(
        json.dumps({**s.to_dict(), "backend": backend_name, "sample_rate": sr}, indent=2),
        encoding="utf-8",
    )
    return meta


# Measured on a Quadro T2000 (4 GB), Windows 11: (track seconds, wall seconds), single run each.
_ACESTEP_POINTS = [(0, 15), (30, 28 + 15), (120, 109 + 15), (240, 454 + 15), (300, 995 + 15)]


def estimate_seconds(backend_name: str, duration: float) -> float:
    """Very rough wall-clock estimate, used only for a progress bar. Hardware dependent."""
    if backend_name == "synth":
        return 1 + duration * 0.025
    if backend_name == "musicgen":
        return 15 + duration * 2.6
    xs, ys = zip(*_ACESTEP_POINTS)
    if duration <= xs[-1]:
        return float(np.interp(duration, xs, ys))
    slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])  # beyond measured range: extrapolate
    return float(ys[-1] + (duration - xs[-1]) * slope)


def available_backends() -> dict[str, bool]:
    """Which backends look usable on this machine (detection only, nothing is loaded)."""
    acestep = False
    try:
        from .backends.acestep import _find_home, _python_in

        acestep = _python_in(_find_home()).exists()
    except Exception:  # noqa: BLE001 - detection only
        pass
    musicgen = bool(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"))
    return {"acestep": acestep, "musicgen": musicgen, "synth": True}


def default_backend() -> str:
    """Best backend that looks usable on this machine: acestep, then musicgen, then synth."""
    try:
        from .backends.acestep import _find_home, _python_in

        if _python_in(_find_home()).exists():
            return "acestep"
    except Exception:  # noqa: BLE001 - detection only
        pass
    if importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"):
        return "musicgen"
    return "synth"
