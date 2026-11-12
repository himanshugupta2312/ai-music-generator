"""Shift generated audio so concert A matches a requested frequency.

Generative models output roughly A=440. Resampling by 440/target raises pitch by
target/440 (tempo changes by the same small factor, ~2% for 450 Hz). This is a
simple linear-interpolation resample: fine for shifts under about a semitone.
"""

from __future__ import annotations

import numpy as np


def retune(audio: np.ndarray, target_hz: float, base_hz: float = 440.0) -> np.ndarray:
    """audio is (samples, channels). Returns audio whose A4 sits at target_hz."""
    if abs(target_hz - base_hz) < 1e-6:
        return audio
    ratio = target_hz / base_hz            # >1 raises pitch and shortens the clip
    n_in = audio.shape[0]
    n_out = int(n_in / ratio)
    src_pos = np.arange(n_out) * ratio
    grid = np.arange(n_in)
    out = np.stack(
        [np.interp(src_pos, grid, audio[:, c]) for c in range(audio.shape[1])], axis=1
    )
    return out.astype(np.float32)
