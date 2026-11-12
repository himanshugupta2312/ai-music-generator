from __future__ import annotations

from .base import Backend


def get_backend(name: str) -> Backend:
    """Import lazily so heavy dependencies are only needed for the chosen backend."""
    if name == "synth":
        from .synth import SynthBackend
        return SynthBackend()
    if name == "musicgen":
        from .musicgen import MusicGenBackend
        return MusicGenBackend()
    if name == "acestep":
        from .acestep import AceStepBackend
        return AceStepBackend()
    raise ValueError(f"Unknown backend '{name}'. Choose: synth, musicgen, acestep")


BACKENDS = ("synth", "musicgen", "acestep")
