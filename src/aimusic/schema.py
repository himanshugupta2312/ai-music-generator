"""Structured settings produced by the prompt parser and consumed by backends."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Settings:
    style_prompt: str = "ambient, calm, soft piano and pads"
    lyrics: str = "[instrumental]"
    duration: float = 60.0          # seconds
    tempo: int | None = None        # bpm hint
    seed: int | None = None
    tuning_hz: float = 440.0        # concert pitch of A4; applied as a post-process shift
    soundscapes: list[str] = field(default_factory=list)   # e.g. ["rain"]
    soundscape_level: float = 0.35  # 0..1 mix level for ambience
    original_prompt: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
