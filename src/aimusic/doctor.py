"""`aimusic doctor`: check the environment and print exact fixes."""

from __future__ import annotations

import importlib.util
import shutil
import sys


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def run() -> int:
    problems = 0

    def line(ok: bool, label: str, fix: str = "") -> None:
        nonlocal problems
        print(f"  [{'ok' if ok else '!!'}] {label}" + (f"\n        fix: {fix}" if not ok and fix else ""))
        if not ok:
            problems += 1

    print("aimusic doctor\n")
    v = sys.version_info
    line(v >= (3, 10), f"Python {v.major}.{v.minor}", "install Python 3.10+ (3.11 for ACE-Step)")
    line(_has("numpy"), "numpy", "pip install numpy")
    line(_has("soundfile"), "soundfile", "pip install soundfile")
    line(shutil.which("ffmpeg") is not None, "ffmpeg on PATH (needed for MP3 export)",
         "install ffmpeg and add it to PATH (e.g. winget install ffmpeg)")

    print("\nOptional: neural backends")
    if _has("torch"):
        import torch
        cuda = torch.cuda.is_available()
        print(f"  [ok] torch {torch.__version__}")
        if cuda:
            props = torch.cuda.get_device_properties(0)
            print(f"  [ok] CUDA GPU: {props.name}, {props.total_memory / 2**30:.1f} GB VRAM")
        else:
            print("  [--] CUDA not available (CPU only; neural models will be slow).\n"
                  "        If you have an NVIDIA GPU, install the CUDA build:\n"
                  "        pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126")
    else:
        print("  [--] torch not installed (only the 'synth' backend will work).\n"
              "        for MusicGen: pip install 'aimusic[musicgen]'")
    print(f"  [{'ok' if _has('transformers') else '--'}] transformers")

    print("\n" + ("All required checks passed." if not problems else f"{problems} problem(s) found."))
    return 1 if problems else 0
