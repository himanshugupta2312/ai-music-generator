"""CPU-only, genre-aware procedural music generator.

No ML model, no GPU, no downloads. It is a small hand-written composer: it reads the
style from the prompt (punk, lofi, hip hop, trap, jazz, cinematic, sleep, ambient or a
punk + hip hop mash-up), picks chords, a groove and instruments for it, and renders
everything with simple numpy synthesis (additive tones, noise drums, a little reverb).

It sounds like a toy synth, not like a trained model, but it is instant and the genres
are clearly different. The seed changes the key, the melody and small variations.
"""

from __future__ import annotations

import numpy as np

from ..schema import Settings
from .base import Backend

SR = 32000

# name -> (default bpm, keywords that select it, in priority order is defined in detect_style)
_BPM = {"ambient": 60, "sleep": 50, "lofi": 80, "hiphop": 90, "trap": 140, "jazz": 110,
        "punk": 165, "cinematic": 72, "mashup": 95}


def detect_style(s: Settings) -> str:
    t = f"{s.original_prompt} {s.style_prompt}".lower()
    has = lambda *k: any(x in t for x in k)  # noqa: E731
    if has("punk") and has("hip hop", "boom bap", "trap"):
        return "mashup"
    if has("punk"):
        return "punk"
    if has("trap"):
        return "trap"
    if has("lofi", "lo-fi"):
        return "lofi"
    if has("hip hop", "boom bap"):
        return "hiphop"
    if has("jazz"):
        return "jazz"
    if has("cinematic", "epic", "trailer", "orchestral"):
        return "cinematic"
    if has("sleep", "lullab"):
        return "sleep"
    return "ambient"


def _hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


# ---- instruments (each returns a mono float array) ---------------------------------------
def _t(d: float) -> np.ndarray:
    return (np.arange(int(d * SR), dtype=np.float32) / np.float32(SR))


def _pad(f, d, atk=0.6, detune=0.003):
    t = _t(d)
    x = np.zeros_like(t)
    for k, det in enumerate((-detune, detune)):          # two slightly detuned voices
        for h in (1, 2, 3):
            x += np.sin(np.float32(2 * np.pi * f * (1 + det) * h) * t + np.float32(k)) * np.float32(h ** -1.6)
    env = np.minimum(1, t / atk) * np.minimum(1, (d - t) / (atk * 1.2 + 1e-6))
    return x * np.clip(env, 0, 1) ** 1.4 * 0.22


def _epiano(f, d):
    t = _t(d)
    x = np.sin(2 * np.pi * f * t) + 0.45 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t * 7) + 0.12 * np.sin(2 * np.pi * 3 * f * t) * np.exp(-t * 11)
    x *= np.exp(-t * 2.2) * (1 + 0.06 * np.sin(2 * np.pi * 5 * t))
    return x * np.minimum(1, t / 0.004) * 0.5


def _pluck(f, d, decay=3.0):
    t = _t(d)
    x = sum(np.sin(2 * np.pi * k * f * t) * k ** -1.4 * np.exp(-t * (decay + k)) for k in range(1, 6))
    return x * np.minimum(1, t / 0.003) * 0.7


def _bass(f, d):
    t = _t(d)
    return (np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * 2 * f * t)) * np.exp(-t * 3.2) * np.minimum(1, t / 0.004) * 0.8


def _s808(f, d):
    t = _t(d)
    fr = f * (1 + 0.9 * np.exp(-t * 45))
    return np.sin(2 * np.pi * np.cumsum(fr, dtype=np.float64) / SR) * np.exp(-t * 1.6) * np.minimum(1, t / 0.003) * 0.9


def _kick(d=0.3):
    t = _t(d)
    fr = 42 + 110 * np.exp(-t * 38)
    return np.sin(2 * np.pi * np.cumsum(fr, dtype=np.float64) / SR) * np.exp(-t * 11) * 1.0


def _snare(rng, d=0.22):
    t = _t(d)
    return (rng.standard_normal(len(t)) * np.exp(-t * 24) * 0.7 + np.sin(2 * np.pi * 190 * t) * np.exp(-t * 28) * 0.5) * 0.8


def _hat(rng, d=0.06, open_=False):
    d = 0.35 if open_ else d
    t = _t(d)
    n = np.diff(rng.standard_normal(len(t) + 1))
    return n * np.exp(-t * (14 if open_ else 70)) * 0.35


def _crunch(f, d, mute=True):
    t = _t(d)
    x = sum(np.sin(2 * np.pi * k * f * t) / k for k in range(1, 7))
    x = np.tanh(2.8 * x)
    env = np.exp(-t * (13 if mute else 2.5)) * np.minimum(1, t / 0.002)
    return x * env * 0.35


def _strings(f, d):
    t = _t(d)
    vib = 1 + 0.004 * np.sin(2 * np.pi * 5.2 * t)
    x = sum(np.sin(2 * np.pi * f * vib * k * t) / k ** 1.3 for k in range(1, 6))
    env = np.minimum(1, t / (0.35 * d + 0.2)) * np.minimum(1, (d - t) / 0.4 + 0.0)
    return x * np.clip(env, 0, 1) * 0.2


def _boom(rng, d=1.6):
    t = _t(d)
    return (np.sin(2 * np.pi * 38 * t) * 1.0 + np.convolve(rng.standard_normal(len(t)), np.ones(40) / 40, "same") * 3) * np.exp(-t * 2.2) * 0.8


def _riser(rng, d):
    t = _t(d)
    n = np.convolve(rng.standard_normal(len(t)), np.ones(8) / 8, "same")
    return n * (t / d) ** 2.4 * 0.35


class _Mix:
    """Dry and 'wet' buses; the wet bus gets a small stereo reverb at the end."""

    def __init__(self, n: int) -> None:
        self.dry = np.zeros((n, 2), np.float32)
        self.wet = np.zeros((n, 2), np.float32)

    def add(self, t0: float, sig: np.ndarray, gain=1.0, pan=0.0, wet=False) -> None:
        buf = self.wet if wet else self.dry
        i = int(t0 * SR)
        if i < 0 or i >= len(buf):
            return
        j = min(len(buf), i + len(sig))
        seg = sig[: j - i].astype(np.float32)
        ang = (pan + 1) * np.pi / 4
        buf[i:j, 0] += seg * gain * np.cos(ang)
        buf[i:j, 1] += seg * gain * np.sin(ang)

    def render(self) -> np.ndarray:
        out = self.dry.copy()
        w = self.wet
        out += w * 0.7
        for delay, g in ((0.037, 0.32), (0.071, 0.26), (0.113, 0.2), (0.179, 0.15), (0.271, 0.1)):
            d = int(delay * SR)
            out[d:, 0] += w[:-d, 1] * g          # cross-feed left/right for width
            out[d:, 1] += w[:-d, 0] * g
        return out


# ---- chord material ----------------------------------------------------------------------
_M7, _D7, _MAJ7 = (0, 3, 7, 10), (0, 4, 7, 10), (0, 4, 7, 11)
_MIN, _MAJ = (0, 3, 7), (0, 4, 7)


def _progression(style: str, major: bool):
    """List of (root offset in semitones, chord intervals) per bar."""
    if style in ("lofi", "jazz"):
        return [(2, _M7), (7, _D7), (0, _MAJ7), (9, _M7)]                    # ii - V - I - vi
    if style in ("punk", "mashup"):
        return [(0, _MAJ), (7, _MAJ), (9, _MIN), (5, _MAJ)] if major else [(0, _MIN), (8, _MAJ), (10, _MAJ), (7, _MIN)]
    if style == "trap":
        return [(0, _MIN), (8, _MAJ), (10, _MAJ), (0, _MIN)]
    if style == "cinematic":
        return [(0, _MIN), (8, _MAJ), (3, _MAJ), (10, _MAJ)]
    if style == "hiphop":
        return [(0, _M7), (8, _MAJ7), (5, _M7), (7, _D7)] if not major else [(0, _MAJ7), (9, _M7), (5, _MAJ7), (7, _D7)]
    # ambient / sleep
    return [(0, _MAJ7), (9, _M7), (5, _MAJ7), (7, _D7)] if major else [(0, _M7), (8, _MAJ7), (3, _MAJ7), (10, _D7)]


def _voicing(root: float, ch, octave=0, spread=False):
    notes = [root + 12 * octave + i for i in ch]
    if spread and len(notes) > 3:
        notes[1] += 12
    return notes


class SynthBackend(Backend):
    name = "synth"
    sample_rate = SR

    def generate(self, s: Settings) -> np.ndarray:
        rng = np.random.default_rng(s.seed)
        style = detect_style(s)
        text = f"{s.original_prompt} {s.style_prompt}".lower()
        major = any(k in text for k in ("happy", "upbeat", "bright", "major"))
        bpm = float(s.tempo or _BPM[style])
        beat = 60.0 / bpm
        bar = 4 * beat
        total_s = float(s.duration)
        n = int(total_s * SR)
        mix = _Mix(n)
        root = 48 + int(rng.integers(-3, 4))                      # key from the seed (around C3)
        prog = _progression(style, major)
        n_bars = int(total_s / bar) + 2
        # pentatonic scale for melody notes
        pent = [0, 3, 5, 7, 10] if not major else [0, 2, 4, 7, 9]
        swing = {"lofi": 0.2, "jazz": 0.3, "hiphop": 0.12}.get(style, 0.0)

        def step_t(b, step):                                       # 8th-note grid with swing
            return b * bar + step * beat / 2 + (swing * beat / 2 if step % 2 else 0.0)

        for b in range(n_bars):
            cr, ch = prog[b % len(prog)]
            croot = root + cr
            prog_pos = b / max(1, n_bars - 1)
            if style in ("ambient", "sleep"):
                soft = 0.6 if style == "sleep" else 1.0
                for i, note in enumerate(_voicing(croot + 12, ch, spread=True)):
                    mix.add(b * bar, _pad(_hz(note), bar * 1.1, atk=bar * 0.35), 0.8 * soft, pan=(-0.5 + i * 0.35), wet=True)
                mix.add(b * bar, _pad(_hz(croot - 12), bar * 1.05, atk=bar * 0.4), 0.7 * soft)
                for _ in range(2 if style == "ambient" else 1):
                    if rng.random() < 0.55:
                        nt = croot + 24 + int(rng.choice(pent))
                        mix.add(b * bar + float(rng.random()) * bar * 0.85, _pluck(_hz(nt), 2.4, 1.6), 0.5 * soft,
                                pan=float(rng.uniform(-0.8, 0.8)), wet=True)

            elif style in ("lofi", "hiphop", "trap"):
                trap = style == "trap"
                for st in ((0, 3) if not trap else (0, 6)):
                    for i, note in enumerate(_voicing(croot + 12, ch)):
                        mix.add(step_t(b, st) + i * 0.012, _epiano(_hz(note), beat * (2.2 if not trap else 3)), 0.32 if not trap else 0.28,
                                pan=-0.35 + i * 0.2, wet=True)
                if trap:
                    mix.add(step_t(b, 0), _s808(_hz(croot - 12), beat * 3.2), 0.95)
                    mix.add(step_t(b, 6), _s808(_hz(croot - 12 + (7 if rng.random() < .4 else 0)), beat * 1.2), 0.8)
                    for st in range(8):
                        for sub in range(2):
                            roll = rng.random() < 0.18 and st % 2 == 1
                            t0 = step_t(b, st) + sub * beat / 4
                            mix.add(t0, _hat(rng), 0.45 if sub == 0 else 0.25, pan=0.3)
                            if roll:
                                mix.add(t0 + beat / 8, _hat(rng), 0.2, pan=0.3)
                    mix.add(step_t(b, 0), _kick(), 1.0)
                    mix.add(step_t(b, 5), _kick(), 0.9 if rng.random() < .6 else 0.0)
                    mix.add(step_t(b, 4), _snare(rng), 0.9)                          # half-time snare on beat 3
                else:
                    for st in (0, 3, 5):
                        mix.add(step_t(b, st), _kick(), 0.95 if st == 0 else 0.7)
                    for st in (2, 6):
                        mix.add(step_t(b, st), _snare(rng), 0.7 if style == "lofi" else 0.85)
                    for st in range(8):
                        mix.add(step_t(b, st), _hat(rng), 0.32 if st % 2 == 0 else 0.18, pan=0.3)
                    mix.add(step_t(b, 0), _bass(_hz(croot - 12), beat * 1.6), 0.9)
                    mix.add(step_t(b, 5), _bass(_hz(croot - 12 + 7), beat * 1.2), 0.7)
                for st in range(8):
                    if rng.random() < (0.3 if style == "lofi" else 0.2):
                        mix.add(step_t(b, st), _epiano(_hz(croot + 24 + int(rng.choice(pent))), beat * 2), 0.22,
                                pan=float(rng.uniform(-0.6, 0.6)), wet=True)

            elif style == "jazz":
                walk = [croot - 12, croot - 12 + ch[1], croot - 12 + ch[2], croot - 12 + ch[3]]
                nxt = prog[(b + 1) % len(prog)][0] + root - 12
                walk[3] = nxt - 1 if rng.random() < 0.6 else nxt + 1
                for k in range(4):
                    mix.add(b * bar + k * beat, _bass(_hz(walk[k]), beat * 0.95), 0.85)
                    mix.add(b * bar + k * beat, _hat(rng, open_=True) * 0.5, 0.5, pan=0.4)           # ride
                    if k in (1, 3):
                        mix.add(b * bar + k * beat + beat * (0.5 + swing / 2), _hat(rng), 0.45, pan=0.4)
                for st in (1, 4, 6):
                    if rng.random() < 0.75:
                        for i, note in enumerate(_voicing(croot + 12, ch)):
                            mix.add(step_t(b, st) + i * 0.01, _epiano(_hz(note), beat * 1.4), 0.3, pan=-0.3 + i * 0.2, wet=True)
                if rng.random() < 0.5:
                    mix.add(step_t(b, 5), _snare(rng, 0.12), 0.2)
                if rng.random() < 0.6:
                    mix.add(step_t(b, int(rng.integers(0, 8))), _pluck(_hz(croot + 24 + int(rng.choice(pent))), 1.6, 2.5), 0.3,
                            pan=float(rng.uniform(-0.6, 0.6)), wet=True)

            elif style in ("punk", "mashup"):
                p5 = [croot, croot + 7, croot + 12]
                for st in range(8):
                    long_ = st in (0, 4)
                    for note in p5:
                        mix.add(step_t(b, st), _crunch(_hz(note + 12), beat * (0.9 if long_ else 0.3), mute=not long_), 0.55 if long_ else 0.4,
                                pan=-0.6 if note != p5[1] else 0.6)
                    mix.add(step_t(b, st), _bass(_hz(croot - 12), beat * 0.45), 0.8)
                if style == "punk":
                    for st in (0, 3, 4):
                        mix.add(step_t(b, st), _kick(), 1.0)
                    for st in (2, 6):
                        mix.add(step_t(b, st), _snare(rng), 0.95)
                    for st in range(8):
                        mix.add(step_t(b, st), _hat(rng), 0.35, pan=0.3)
                else:   # mash-up: punk guitars over boom-bap drums and an 808
                    for st in (0, 3, 5):
                        mix.add(step_t(b, st), _kick(), 1.0)
                    for st in (2, 6):
                        mix.add(step_t(b, st), _snare(rng), 0.95)
                    for st in range(8):
                        mix.add(step_t(b, st), _hat(rng), 0.3, pan=0.3)
                    mix.add(step_t(b, 0), _s808(_hz(croot - 12), beat * 1.6), 0.8)
                if b % 4 == 0:
                    mix.add(b * bar, _hat(rng, open_=True) * 2.2, 0.7, pan=-0.2)      # crash

            elif style == "cinematic":
                inten = 0.35 + 0.65 * min(1.0, prog_pos * 1.5 + b / 24)
                for i, note in enumerate(_voicing(croot, ch, spread=True)):
                    mix.add(b * bar, _strings(_hz(note + 12), bar * 1.05), inten * 0.9, pan=-0.6 + i * 0.4, wet=True)
                mix.add(b * bar, _pad(_hz(croot - 24), bar * 1.1, atk=bar * 0.3), 0.8)
                if b >= 2:
                    for st in range(8):
                        mix.add(step_t(b, st), _pluck(_hz(croot + (0 if st % 4 else 7)), beat * 0.5, 6), 0.3 * inten, pan=0.4 if st % 2 else -0.4, wet=True)
                mix.add(b * bar, _kick(0.5), 0.9 * inten)
                if b % 4 == 3:
                    mix.add(b * bar + bar - min(bar, 3.0), _riser(rng, min(bar, 3.0)), inten)
                if b % 4 == 0 and b > 0:
                    mix.add(b * bar, _boom(rng), 1.1 * inten)

        out = mix.render()[:n]
        if style in ("lofi", "hiphop"):                            # vinyl crackle
            idx = rng.integers(0, n, size=int(total_s * 18))
            out[idx, 0] += rng.uniform(0.02, 0.12, size=len(idx)).astype(np.float32)
            out[idx, 1] += rng.uniform(0.02, 0.12, size=len(idx)).astype(np.float32)
        fade = min(int(SR * 1.2), n // 4)
        if fade > 0:                                               # gentle fade in/out
            ramp = np.linspace(0, 1, fade, dtype=np.float32)[:, None]
            out[:fade] *= ramp
            out[-fade:] *= ramp[::-1]
        peak = float(np.max(np.abs(out))) or 1.0
        return (np.tanh(out / peak * 1.25) * 0.85).astype(np.float32)
