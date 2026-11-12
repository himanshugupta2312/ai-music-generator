"""MusicGen via Hugging Face transformers.

Install: pip install "aimusic[musicgen]"  (use the CUDA build of torch for GPU).
LICENSE NOTE: MusicGen model weights are CC-BY-NC (non-commercial).
"""

from __future__ import annotations

import numpy as np

from ..schema import Settings
from .base import Backend, crossfade_concat

MODEL_ID = "facebook/musicgen-small"
CHUNK_SECONDS = 30      # MusicGen is trained on ~30 s windows
TOKENS_PER_SECOND = 50  # EnCodec frame rate for MusicGen


class MusicGenBackend(Backend):
    name = "musicgen"
    sample_rate = 32000

    def __init__(self, model_id: str = MODEL_ID):
        try:
            import torch
            from transformers import AutoProcessor, MusicgenForConditionalGeneration
        except ImportError as e:
            raise RuntimeError(
                "MusicGen backend needs extra packages: pip install 'aimusic[musicgen]'"
            ) from e
        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = MusicgenForConditionalGeneration.from_pretrained(model_id).to(self.device)
        self.sample_rate = self.model.config.audio_encoder.sampling_rate

    def generate(self, s: Settings) -> np.ndarray:
        torch = self._torch
        if s.seed is not None:
            torch.manual_seed(s.seed)
        remaining = s.duration
        chunks = []
        while remaining > 0:
            secs = min(CHUNK_SECONDS, remaining + 2)  # +2 s for crossfade overlap
            inputs = self.processor(text=[s.style_prompt], padding=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                audio = self.model.generate(
                    **inputs, do_sample=True, guidance_scale=3.0,
                    max_new_tokens=int(secs * TOKENS_PER_SECOND),
                )
            chunks.append(audio[0].cpu().numpy().T.astype(np.float32))  # (samples, ch)
            remaining -= CHUNK_SECONDS - 2
        out = crossfade_concat(chunks, self.sample_rate, fade_s=2.0)
        return out[: int(s.duration * self.sample_rate)]
