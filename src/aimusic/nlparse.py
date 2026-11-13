"""Turn a plain-English request into structured Settings.

Rules handle the reliable parts (duration, instrumental, ambience). An optional
local LLM (Ollama) can enrich the style description; it is never required.
"""

from __future__ import annotations

import json
import re
import urllib.request

from .schema import Settings

_UNITS = {
    "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
    "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
    "s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1,
}
_DURATION_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*-?\s*(hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)\b",
    re.IGNORECASE,
)

# keyword -> style descriptors added to the model prompt
_MOODS = {
    "soothing": ["soothing", "calm", "gentle", "slow tempo"],
    "calm": ["calm", "gentle"],
    "relax": ["relaxing", "calm", "slow tempo"],
    "sleep": ["sleep music", "very slow", "soft pads", "warm"],
    "study": ["focus music", "ambient", "minimal", "steady"],
    "focus": ["focus music", "ambient", "minimal", "steady"],
    "lofi": ["lo-fi", "mellow", "vinyl crackle"],
    "lo-fi": ["lo-fi", "mellow", "vinyl crackle"],
    "forest": ["earthy", "warm", "acoustic", "organic textures"],
    "cinematic": ["cinematic", "orchestral", "emotional"],
    "epic": ["epic", "cinematic", "powerful drums"],
    "dark": ["dark", "moody", "low synth"],
    "happy": ["upbeat", "bright", "major key"],
    "sad": ["melancholic", "minor key", "slow piano"],
    "jazz": ["jazz", "soft piano", "brushed drums"],
    "punk": ["punk rock", "distorted electric guitars", "fast power chords", "raw", "energetic"],
    "hip": ["hip hop", "boom bap drum beat", "808 bass", "sampled loops"],  # "hip hop"/"hip-hop"
    "trap": ["trap", "808 bass", "hi-hat rolls"],
    "piano": ["solo piano"],
    "acoustic": ["acoustic guitar", "warm"],
    "guitar": ["guitar"],  # neutral: don't force "acoustic" onto punk/rock prompts
}

_SOUNDSCAPES = {
    "rain": ("rain", "raining", "rainy", "rainfall"),
    "wind": ("wind", "windy", "breeze"),
}

_VOCAL_HINTS = ("lyrics", "vocals", "sing", "song with words", "sung")
_TEMPO_RE = re.compile(r"(\d{2,3})\s*bpm", re.IGNORECASE)
# "450hz" / "A=450 Hz" -> tuning. Only plausible concert-pitch values are accepted.
_TUNING_RE = re.compile(r"(\d{3}(?:\.\d+)?)\s*hz", re.IGNORECASE)
_THEME_RE = re.compile(r"theme(?:\s+(?:could be|is|of))?\s*[:\-]?\s*(.+)$", re.IGNORECASE)


def parse_duration(text: str) -> float | None:
    m = _DURATION_RE.search(text)
    if not m:
        return None
    return float(m.group(1)) * _UNITS[m.group(2).lower()]


def parse_prompt(text: str) -> Settings:
    low = text.lower()
    s = Settings(original_prompt=text)

    dur = parse_duration(text)
    if dur:
        s.duration = dur

    m = _TEMPO_RE.search(text)
    if m:
        s.tempo = int(m.group(1))

    m = _TUNING_RE.search(text)
    if m and 400 <= float(m.group(1)) <= 480:
        s.tuning_hz = float(m.group(1))

    words = set(re.findall(r"[a-z\-]+", low))
    descriptors: list[str] = []
    for key, desc in _MOODS.items():
        # prefix match so "studying"/"relaxing" hit "study"/"relax"
        if any(w.startswith(key) for w in words):
            for d in desc:
                if d not in descriptors:
                    descriptors.append(d)

    for name, triggers in _SOUNDSCAPES.items():
        if any(t in words for t in triggers):
            s.soundscapes.append(name)

    theme = _THEME_RE.search(text)
    if theme:
        # keep the theme words as atmosphere hints for the model
        descriptors.append(theme.group(1).strip().rstrip("."))

    if descriptors:
        s.style_prompt = ", ".join(descriptors)
    if s.tempo:
        s.style_prompt += f", {s.tempo} bpm"

    if any(h in low for h in _VOCAL_HINTS):
        s.lyrics = ""  # user should supply lyrics; empty means "let model improvise"
    return s


def enrich_with_ollama(
    s: Settings, model: str = "llama3.2", host: str = "http://localhost:11434"
) -> Settings:
    """Ask a local Ollama model for a better style prompt. Fails soft."""
    body = {
        "model": model,
        "stream": False,
        "format": "json",
        "prompt": (
            "Rewrite this music request as a short comma-separated list of style tags "
            "(genre, mood, instruments, tempo, texture) for a text-to-music model. "
            'Reply as JSON: {"style_prompt": "..."}. Request: ' + s.original_prompt
        ),
    }
    try:
        req = urllib.request.Request(
            f"{host}/api/generate",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            reply = json.loads(json.loads(r.read())["response"])
        style = str(reply.get("style_prompt", "")).strip()
        if style:
            s.style_prompt = style
    except Exception as e:  # noqa: BLE001 - optional feature, never fatal
        print(f"[note] LLM enrichment skipped ({type(e).__name__}); using rule-based prompt.")
    return s
