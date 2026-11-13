import numpy as np
import pytest

from aimusic.backends.synth import SynthBackend, detect_style
from aimusic.pipeline import build_settings

PROMPTS = {
    "ambient": "soothing ambient music",
    "sleep": "sleep music with soft piano and wind",
    "lofi": "lofi hip hop beats, mellow, 85 bpm",
    "hiphop": "hip hop beats at 90 bpm",
    "trap": "dark trap beats",
    "jazz": "mellow jazz with soft piano, 70 bpm",
    "punk": "punk rock with distorted guitars and fast drums",
    "cinematic": "epic cinematic music",
    "mashup": "soundtrack with hip hop beats and punk music at 450hz",
}


@pytest.mark.parametrize("style,prompt", PROMPTS.items())
def test_style_is_detected(style, prompt):
    assert detect_style(build_settings(prompt)) == style


def test_every_style_renders_valid_audio_and_they_differ():
    b = SynthBackend()
    clips = {}
    for style, prompt in PROMPTS.items():
        a = b.generate(build_settings(prompt, duration=8, seed=1))
        assert a.shape == (8 * b.sample_rate, 2) and np.isfinite(a).all()
        assert 0.05 < np.abs(a).max() <= 1.0                       # audible, not clipped
        clips[style] = a
    names = list(clips)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            x, y = clips[names[i]], clips[names[j]]
            assert not np.allclose(x, y), f"{names[i]} and {names[j]} are identical"
    # punk (fast, dense drums) has far more energy changes than a slow ambient pad
    def activity(a):
        m = np.abs(a.mean(axis=1))
        env = m[: len(m) // 800 * 800].reshape(-1, 800).mean(axis=1)
        return float(np.abs(np.diff(env)).mean() / (env.mean() + 1e-9))
    assert activity(clips["punk"]) > 1.5 * activity(clips["ambient"])


def test_seed_is_deterministic_and_changes_the_music():
    b = SynthBackend()
    s = lambda seed: b.generate(build_settings("mellow jazz", duration=6, seed=seed))  # noqa: E731
    assert np.array_equal(s(4), s(4))
    assert not np.allclose(s(4), s(5))


def test_tempo_is_used_for_the_groove():
    b = SynthBackend()
    slow = b.generate(build_settings("lofi hip hop beats, 70 bpm", duration=8, seed=2))
    fast = b.generate(build_settings("lofi hip hop beats, 120 bpm", duration=8, seed=2))
    assert not np.allclose(slow, fast)
