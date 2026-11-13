"""One-click prompt presets ("vibes") and the Surprise me prompt generator.

Colours (c1, c2) drive the palette of the 3D scene in the web UI.
"""

from __future__ import annotations

import random

# id, emoji, name, short description, prompt, colour 1, colour 2
VIBES = [
    {"id": "study", "emoji": "📚", "name": "Study time", "sub": "calm piano + rain",
     "prompt": "1 min calm soothing piano music with rain for studying", "c1": "#38bdf8", "c2": "#6366f1"},
    {"id": "workout", "emoji": "🏋️", "name": "Workout", "sub": "punk, fast drums",
     "prompt": "45 sec punk rock with distorted guitars and fast drums", "c1": "#ef4444", "c2": "#f59e0b"},
    {"id": "lofi", "emoji": "☕", "name": "Lofi café", "sub": "mellow beats, 85 bpm",
     "prompt": "45 sec lofi hip hop beats, mellow, 85 bpm", "c1": "#f59e0b", "c2": "#a855f7"},
    {"id": "trailer", "emoji": "🎬", "name": "Movie trailer", "sub": "epic + cinematic",
     "prompt": "45 sec epic cinematic music", "c1": "#fbbf24", "c2": "#3b82f6"},
    {"id": "sleep", "emoji": "🌙", "name": "Sleepy time", "sub": "soft piano + wind",
     "prompt": "2 min sleep music with soft piano and wind", "c1": "#818cf8", "c2": "#4338ca"},
    {"id": "jazz", "emoji": "🎷", "name": "Jazz night", "sub": "mellow jazz, 70 bpm",
     "prompt": "45 sec mellow jazz with soft piano, 70 bpm", "c1": "#fb923c", "c2": "#be185d"},
    {"id": "forest", "emoji": "🌲", "name": "Forest floor", "sub": "earthy + rain",
     "prompt": "1 min soothing music, theme: sitting on a forest floor, with rain", "c1": "#22c55e", "c2": "#14b8a6"},
    {"id": "mashup", "emoji": "🎛️", "name": "Genre mash-up", "sub": "hip hop + punk, A=450",
     "prompt": "45 sec soundtrack with hip hop beats and punk music at 450hz", "c1": "#ec4899", "c2": "#8b5cf6"},
]

_LENGTHS = ["30 sec", "45 sec", "1 min", "90 sec"]
_STYLES = [
    "calm piano", "lofi hip hop beats", "punk rock", "epic cinematic music", "mellow jazz",
    "dark trap beats", "happy upbeat tune", "soothing ambient music", "sad piano",
]
_EXTRAS = ["", "", " with rain", " with wind", " for studying", " for a rainy night"]
_TEMPOS = ["", "", ", 70 bpm", ", 90 bpm", ", 120 bpm"]


def surprise_prompt(rng: random.Random | None = None) -> str:
    """A random, always-valid prompt for the Surprise me button."""
    r = rng or random
    return f"{r.choice(_LENGTHS)} {r.choice(_STYLES)}{r.choice(_EXTRAS)}{r.choice(_TEMPOS)}"
