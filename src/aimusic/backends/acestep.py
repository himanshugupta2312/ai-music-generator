"""ACE-Step backend (placeholder).

Not implemented yet: I have not verified ACE-Step 1.5's official Python API or
dependency pins. Next step: read github.com/ace-step/ACE-Step, then wire its
inference entry point here (style prompt + lyrics + duration + seed).
"""

from __future__ import annotations

import numpy as np

from ..schema import Settings
from .base import Backend


class AceStepBackend(Backend):
    name = "acestep"
    sample_rate = 44100

    def generate(self, s: Settings) -> np.ndarray:
        raise NotImplementedError(
            "ACE-Step backend is not wired up yet. Use --backend synth or musicgen."
        )
