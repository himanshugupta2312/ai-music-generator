"""CPU-only procedural ambient generator.

No ML model, no GPU, no downloads. Lower fidelity than neural backends, but it
lets the whole pipeline (parse -> generate -> mix rain -> export) run anywhere
and makes a good test backend.
"""

from __future__ import annotations

import numpy as np

from ..schema import Settings
from .base import Backend

# A-minor-ish progression (MIDI root notes), each chord held for a few seconds.
_PROGRESSION = [
    (57, [0, 3, 7, 10]),   # Am7
    (53, [0, 4, 7, 11]),   # Fmaj7
    (48, [0, 4, 7, 11]),   # Cmaj7
    (55, [0, 4, 7, 10]),   # G7
]


def _midi_hz(n: float) -> float:
    return 440.0 * 2 ** ((n - 69) / 12)


class SynthBackend(Backend):
    name = "synth"
    sample_rate = 32000

    def generate(self, s: Settings) -> np.ndarray:
        sr = self.sample_rate
        rng = np.random.default_rng(s.seed)
        total = int(s.duration * sr)
        out = np.zeros((total, 2), dtype=np.float32)
        chord_len = int(sr * (60.0 / (s.tempo or 60)) * 8)  # 8 beats per chord
        chord_len = max(chord_len, sr * 4)

        pos, i = 0, 0
        while pos < total:
            root, intervals = _PROGRESSION[i % len(_PROGRESSION)]
            n = min(chord_len + sr * 2, total - pos)  # overlap tails
            t = np.arange(n) / sr
            env = np.minimum(1.0, t / 2.0) * np.minimum(1.0, (n / sr - t) / 2.0)
            env = np.clip(env, 0, 1) ** 1.5
            for k, iv in enumerate(intervals):
                for octave in (0, 12):
                    f = _midi_hz(root + iv + octave)
                    detune = rng.uniform(-0.15, 0.15)
                    phase = rng.uniform(0, 2 * np.pi)
                    tone = np.sin(2 * np.pi * (f + detune) * t + phase)
                    tone += 0.25 * np.sin(2 * np.pi * 2 * f * t + phase)
                    amp = 0.06 if octave == 0 else 0.03
                    pan = 0.5 + 0.35 * np.sin(k + octave)
                    out[pos:pos + n, 0] += (tone * env * amp * (1 - pan)).astype(np.float32)
                    out[pos:pos + n, 1] += (tone * env * amp * pan).astype(np.float32)
            pos += chord_len
            i += 1

        peak = float(np.max(np.abs(out))) or 1.0
        return (out / peak * 0.8).astype(np.float32)
