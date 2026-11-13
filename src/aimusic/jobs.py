"""Background generation jobs for the web UI.

One job at a time: the GPU is the bottleneck, and a 4 GB card cannot hold two models.
"""

from __future__ import annotations

import gc
import random
import re
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from .backends import get_backend
from .pipeline import build_settings, estimate_seconds, render, save, validate
from .schema import Settings


class Busy(Exception):
    """Raised when a new job is requested while another is still running."""


class _Cancelled(Exception):
    pass


def _free_gpu() -> None:
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


class Job:
    def __init__(self, prompt: str, backend_name: str, settings: Settings,
                 warnings: list[str], out_dir: Path) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.prompt = prompt
        self.backend_name = backend_name
        self.settings = settings
        self.warnings = warnings
        self.out_dir = out_dir
        self.estimate = estimate_seconds(backend_name, settings.duration)
        self.state = "running"          # running | done | error | cancelled
        self.started = time.time()
        self.finished: float | None = None
        self.result: dict | None = None
        self.error: str | None = None
        self.cancel_requested = False
        self._backend = None
        self._thread = threading.Thread(target=self._work, daemon=True)

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        self._thread.start()

    def alive(self) -> bool:
        return self._thread.is_alive()

    def cancel(self) -> None:
        """Stop the job. ACE-Step is really interrupted; other backends finish in the
        background but their result is discarded."""
        self.cancel_requested = True
        backend = self._backend
        if backend is not None:
            backend.cancel()
        if self.state == "running":
            self.state = "cancelled"

    def _work(self) -> None:
        backend = None
        try:
            backend = get_backend(self.backend_name)
            self._backend = backend
            if self.cancel_requested:
                raise _Cancelled()
            audio = render(backend, self.settings)
            if self.cancel_requested:
                raise _Cancelled()
            slug = re.sub(r"[^a-z0-9]+", "-", self.prompt.lower()).strip("-")[:30] or "track"
            out = self.out_dir / f"{datetime.now():%Y%m%d-%H%M%S}-{slug}.wav"
            save(audio, backend.sample_rate, out, self.settings, backend.name)
            self.result = {
                "name": out.name,
                "audio_url": f"/output/{out.name}",
                "settings": self.settings.to_dict(),
                "backend": backend.name,
                "label": f"{datetime.now():%H:%M} · {self.prompt.strip()[:48]}",
                "took_s": round(time.time() - self.started, 1),
            }
            if self.state == "running":
                self.state = "done"
        except _Cancelled:
            self.state = "cancelled"
        except BaseException as e:  # noqa: BLE001 - reported to the UI
            if self.cancel_requested:
                self.state = "cancelled"
            else:
                self.error = str(e)[:600] or type(e).__name__
                self.state = "error"
        finally:
            self._backend = None
            del backend
            self.finished = time.time()
            _free_gpu()

    # -- reporting ---------------------------------------------------------
    def snapshot(self) -> dict:
        end = self.finished or time.time()
        return {
            "id": self.id,
            "state": self.state,
            "backend": self.backend_name,
            "elapsed": round(end - self.started, 1),
            "estimate": round(self.estimate, 1),
            "warnings": self.warnings,
            "result": self.result if self.state == "done" else None,
            "error": self.error if self.state == "error" else None,
        }


class JobManager:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.jobs: dict[str, Job] = {}
        self.current: Job | None = None
        self._lock = threading.Lock()

    def start(self, prompt: str, backend_name: str, *, duration=None, seed=None, lyrics=None,
              tuning=None, ambience=True, ambience_level=None) -> Job:
        """Validate and start a job. Raises ValueError (bad input) or Busy."""
        with self._lock:
            if self.current is not None and self.current.alive():
                raise Busy("Another track is still being made. Stop it or wait for it to finish.")
            if not prompt or not prompt.strip():
                raise ValueError("Pick a vibe or describe the music first.")
            s = build_settings(
                prompt, duration=duration, lyrics=lyrics or None, tuning=tuning,
                ambience=ambience, ambience_level=ambience_level,
                seed=random.randint(1, 99999) if seed in (None, "") else int(seed),
            )
            warnings = validate(s, backend_name)
            job = Job(prompt, backend_name, s, warnings, self.out_dir)
            self.jobs[job.id] = job
            self.current = job
            job.start()
            return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def history(self) -> list[dict]:
        done = [j for j in self.jobs.values() if j.state == "done" and j.result]
        done.sort(key=lambda j: j.finished or 0, reverse=True)
        return [j.result for j in done]
