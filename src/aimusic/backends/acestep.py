"""ACE-Step 1.5 backend (single-pass generation, up to 10 minutes).

ACE-Step pins its own PyTorch (2.7.1+cu128 on Windows), which conflicts with the
MusicGen setup, so it lives in a separate virtual environment and is run as a
subprocess through `acestep_worker.py`.

Setup (once):
    git clone https://github.com/ace-step/ACE-Step-1.5
    cd ACE-Step-1.5 && uv sync
Point aimusic at it with the ACESTEP_HOME environment variable. If unset, a sibling
folder named ACE-Step-1.5 next to this repo is used.

Verified on: Windows 11, Quadro T2000 (4 GB), 2 min in 109 s, 4 min in 454 s.
Model weights download (about 7 GB) on first use. ACE-Step is MIT licensed.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from ..schema import Settings
from .base import Backend

WORKER = Path(__file__).with_name("acestep_worker.py")


def _find_home() -> Path:
    env = os.environ.get("ACESTEP_HOME")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[4] / "ACE-Step-1.5"


def _python_in(home: Path) -> Path:
    sub = ("Scripts", "python.exe") if sys.platform == "win32" else ("bin", "python")
    return home.joinpath(".venv", *sub)


class AceStepBackend(Backend):
    name = "acestep"
    sample_rate = 48000

    def __init__(self) -> None:
        self.home = _find_home()
        self.python = _python_in(self.home)
        if not self.python.exists():
            raise RuntimeError(
                f"ACE-Step environment not found at {self.home}.\n"
                "Install it (git clone https://github.com/ace-step/ACE-Step-1.5, then "
                "`uv sync` inside it) or set ACESTEP_HOME to its folder."
            )

    def generate(self, s: Settings) -> np.ndarray:
        import json

        import soundfile as sf

        instrumental = s.lyrics.strip().lower() in ("", "[instrumental]")
        with tempfile.TemporaryDirectory(prefix="aimusic-ace-") as td:
            job = {
                "ace_home": str(self.home),
                "caption": s.style_prompt,
                "lyrics": "[Instrumental]" if instrumental else s.lyrics,
                "instrumental": instrumental,
                "duration": s.duration,
                "seed": s.seed,
                "out_dir": td,
            }
            job_file = Path(td) / "job.json"
            job_file.write_text(json.dumps(job), encoding="utf-8")

            env = os.environ.copy()
            env.setdefault("ACESTEP_GENERATION_TIMEOUT", "3600")
            env["PYTHONIOENCODING"] = "utf-8"
            proc = subprocess.run(
                [str(self.python), "-I", str(WORKER), str(job_file)],
                env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
            )
            lines = proc.stdout.splitlines()
            path = next((l.split(" ", 1)[1].strip() for l in lines if l.startswith("AUDIO ")), None)
            if proc.returncode != 0 or not path:
                tail = "\n".join(l for l in lines if l.startswith(("ERROR", "[worker]")) or "Error" in l)[-1500:]
                raise RuntimeError(f"ACE-Step generation failed (exit {proc.returncode}).\n{tail}")
            audio, sr = sf.read(path, dtype="float32", always_2d=True)
            self.sample_rate = sr
            return audio
