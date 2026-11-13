import threading
import time

import numpy as np
import pytest

from aimusic import pipeline
from aimusic.backends.base import Backend


def test_build_settings_overrides():
    s = pipeline.build_settings("calm piano with rain", duration=90, seed=5, tuning=450, ambience=False)
    assert s.duration == 90 and s.seed == 5 and s.tuning_hz == 450
    assert s.soundscapes == []  # ambience switched off even though the prompt asked for rain
    s2 = pipeline.build_settings("calm piano with rain")
    assert s2.soundscapes == ["rain"] and s2.seed is None


def test_validate_limits_and_warnings():
    s = pipeline.build_settings("calm piano", duration=700)
    with pytest.raises(ValueError, match="10-minute"):
        pipeline.validate(s, "acestep")
    s = pipeline.build_settings("calm piano", duration=2)
    with pytest.raises(ValueError, match="at least"):
        pipeline.validate(s, "synth")
    s = pipeline.build_settings("calm piano", duration=120)
    assert pipeline.validate(s, "acestep") == []
    assert any("continuous" in w for w in pipeline.validate(s, "musicgen"))


def test_estimates_grow_with_length():
    for name in ("synth", "musicgen", "acestep"):
        vals = [pipeline.estimate_seconds(name, d) for d in (10, 60, 240, 600)]
        assert vals == sorted(vals) and vals[0] > 0
    assert pipeline.estimate_seconds("acestep", 300) == pytest.approx(995 + 15)


def test_default_backend_is_valid():
    assert pipeline.default_backend() in ("acestep", "musicgen", "synth")


def test_save_writes_audio_and_json(tmp_path):
    s = pipeline.build_settings("calm piano", duration=5, seed=1)
    audio = np.zeros((4800, 2), dtype=np.float32)
    meta = pipeline.save(audio, 48000, tmp_path / "x.wav", s, "synth")
    assert (tmp_path / "x.wav").exists() and meta.name == "x.json"
    assert '"backend": "synth"' in meta.read_text(encoding="utf-8")


def test_describe_lists_all_fields():
    labels = [l for l, _ in pipeline.describe(pipeline.build_settings("4 min calm rain at 450hz"))]
    assert labels == ["style", "duration", "vocals", "ambience", "tuning", "seed"]


def test_backend_cancel_hook_exists():
    class Slow(Backend):
        def generate(self, s):
            return np.zeros((1, 1), dtype=np.float32)

    Slow().cancel()  # default is a harmless no-op
