from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..schema import Settings


class Backend(ABC):
    name = "base"
    sample_rate = 32000

    @abstractmethod
    def generate(self, s: Settings) -> np.ndarray:
        """Return float32 audio shaped (samples, channels) in [-1, 1]."""

    def cancel(self) -> None:
        """Try to stop a running generate() from another thread. Default: cannot be stopped."""


def crossfade_concat(chunks: list[np.ndarray], sr: int, fade_s: float = 2.0) -> np.ndarray:
    """Join audio chunks with an equal-power-ish linear crossfade."""
    out = chunks[0]
    n = int(sr * fade_s)
    for nxt in chunks[1:]:
        k = min(n, len(out), len(nxt))
        if k <= 0:
            out = np.concatenate([out, nxt])
            continue
        fade = np.linspace(0.0, 1.0, k, dtype=np.float32)[:, None]
        mixed = out[-k:] * (1 - fade) + nxt[:k] * fade
        out = np.concatenate([out[:-k], mixed, nxt[k:]])
    return out
