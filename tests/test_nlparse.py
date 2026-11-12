from aimusic.nlparse import parse_duration, parse_prompt
from aimusic.soundscape import mix_soundscapes
from aimusic.backends.synth import SynthBackend
import numpy as np

PROMPT = (
    "Create 4 min long soothing music with rain in the background and it should be "
    "good for studying and theme could be we're sitting on a forest floor"
)


def test_duration_units():
    assert parse_duration("4 min long") == 240
    assert parse_duration("30 seconds") == 30
    assert parse_duration("2-hour mix") == 7200
    assert parse_duration("no duration here") is None


def test_full_prompt():
    s = parse_prompt(PROMPT)
    assert s.duration == 240
    assert s.soundscapes == ["rain"]
    assert s.lyrics == "[instrumental]"
    assert "soothing" in s.style_prompt
    assert "focus music" in s.style_prompt
    assert "sitting on a forest floor" in s.style_prompt


def test_tempo_and_vocals():
    s = parse_prompt("upbeat jazz at 120 bpm with lyrics")
    assert s.tempo == 120
    assert s.lyrics == ""


def test_tuning_and_genres():
    s = parse_prompt("soundtrack with hip hop beats and punk kind of music at 450hz")
    assert s.tuning_hz == 450
    assert "punk rock" in s.style_prompt and "hip hop" in s.style_prompt
    assert parse_prompt("calm music").tuning_hz == 440
    assert parse_prompt("something at 9000 hz").tuning_hz == 440  # implausible, ignored


def test_retune_shifts_pitch():
    from aimusic.tuning import retune
    sr = 32000
    t = np.arange(sr * 2) / sr
    tone = np.sin(2 * np.pi * 440 * t).astype(np.float32)[:, None]
    out = retune(tone, 450)
    spec = np.abs(np.fft.rfft(out[:, 0]))
    peak_hz = np.fft.rfftfreq(len(out), 1 / sr)[spec.argmax()]
    assert abs(peak_hz - 450) < 1.0
    assert retune(tone, 440) is tone


def test_synth_and_rain_pipeline():
    s = parse_prompt("10 seconds calm music with rain")
    s.duration = 6
    s.seed = 1
    b = SynthBackend()
    audio = b.generate(s)
    assert audio.shape == (6 * b.sample_rate, 2)
    mixed = mix_soundscapes(audio, b.sample_rate, s.soundscapes, 0.35, 1)
    assert mixed.shape == audio.shape
    assert np.isfinite(mixed).all()
    assert np.max(np.abs(mixed)) <= 1.0
    assert not np.allclose(mixed, audio)
