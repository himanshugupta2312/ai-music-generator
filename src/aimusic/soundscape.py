"""Procedural ambience layers, mixed under the music. No bundled audio files,
so there are no sample licensing concerns."""

from __future__ import annotations

import numpy as np


def _band_noise(n: int, sr: int, lo: float, hi: float, rng: np.random.Generator) -> np.ndarray:
    """Noise shaped by an FFT band-pass between lo and hi Hz."""
    spec = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1 / sr)
    mask = np.where((freqs >= lo) & (freqs <= hi), 1.0, 0.0)
    # soften band edges
    mask = np.convolve(mask, np.hanning(101) / np.hanning(101).sum(), mode="same")
    return np.fft.irfft(spec * mask, n)


def rain(n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    base = _band_noise(n, sr, 1500, 9000, rng)                  # steady hiss
    low = _band_noise(n, sr, 300, 1200, rng) * 0.4              # body
    drops = np.zeros(n)
    idx = rng.integers(0, n, size=int(n / sr * 40))             # ~40 drops/sec
    drops[idx] = rng.uniform(0.3, 1.0, size=len(idx))
    drops = np.convolve(drops, np.exp(-np.arange(200) / 25.0), mode="same")
    drops = np.diff(drops, prepend=0)  # sharpen into clicks
    sig = base + low + 0.8 * drops * np.std(base)
    return (sig / (np.max(np.abs(sig)) or 1.0)).astype(np.float32)


def wind(n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    sig = _band_noise(n, sr, 80, 700, rng)
    t = np.arange(n) / sr
    gust = 0.6 + 0.4 * np.sin(2 * np.pi * 0.07 * t + rng.uniform(0, 6.28))
    sig = sig * gust
    return (sig / (np.max(np.abs(sig)) or 1.0)).astype(np.float32)


_LAYERS = {"rain": rain, "wind": wind}


def mix_soundscapes(
    audio: np.ndarray, sr: int, layers: list[str], level: float, seed: int | None
) -> np.ndarray:
    """Add ambience layers under `audio` (samples, channels) at `level` (0..1)."""
    if not layers:
        return audio
    rng = np.random.default_rng(seed)
    n, ch = audio.shape
    out = audio.copy()
    for name in layers:
        fn = _LAYERS.get(name)
        if fn is None:
            continue
        # independent noise per channel gives natural stereo width
        amb = np.stack([fn(n, sr, rng) for _ in range(ch)], axis=1)
        fade = min(int(sr * 3), n // 2)
        env = np.ones(n, dtype=np.float32)
        env[:fade] = np.linspace(0, 1, fade)
        env[-fade:] = np.linspace(1, 0, fade)
        out += amb * env[:, None] * level * 0.6
    peak = float(np.max(np.abs(out))) or 1.0
    return (out / max(peak, 1.0) * (0.95 if peak > 1.0 else 1.0)).astype(np.float32)
