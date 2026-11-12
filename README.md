# aimusic

Describe music in plain English; get an audio file. Runs locally with open-source models.

```
aimusic "Create 4 min long soothing music with rain in the background, good for studying, theme: sitting on a forest floor"
```

The tool shows how it understood your request (duration, style, ambience,
tuning), asks for confirmation, generates, mixes in an ambience layer
(rain/wind), and saves a `.wav` plus a `.json` with every setting used.

> **Early-stage project (v0.1).** One neural backend works. Long tracks (over
> 30 s) are not yet continuous, and the ACE-Step backend is not implemented.
> See [Status](#status) before relying on anything.

## How it works

```
plain-English prompt -> parser -> structured settings -> backend -> ambience mix -> retune -> .wav/.mp3 + .json
```

- **Parser** (rule-based): extracts duration ("4 min"), tempo ("120 bpm"), mood and
  genre keywords, rain/wind, instrumental vs vocals, and tuning ("450hz").
  Optionally a local Ollama model can rewrite the style description (`--llm`).
- **Backends** (swappable): `synth` (CPU, no model), `musicgen`, `acestep` (placeholder).
- **Ambience:** rain and wind are synthesized procedurally, so no sample licensing applies.
- **Tuning:** `--tuning 450` shifts finished audio so concert A sits at 450 Hz.

## Status

| Part | State |
|---|---|
| Plain-English parser | working, rule-based; 6 unit tests pass |
| `synth` backend (CPU, no model, basic ambient pads) | working; for testing and no-GPU use |
| `musicgen` backend (Hugging Face transformers) | working. Tested on Windows 11, Quadro T2000 (4 GB), torch 2.14.1+cu126, transformers 5.22.2. A 45 s track took about 118 s, a 60 s track about 230 s (slower than real time). |
| Tracks over 30 s | **limited**: made of independent 30 s chunks joined by a crossfade, so the music can change character at the joins |
| Rain/wind mixer, `--tuning`, `doctor` | working; the retune was verified on a pure tone, not on full tracks |
| `--llm` (Ollama) enrichment | implemented, **never run** against a live Ollama |
| `acestep` backend | **placeholder**; raises "not wired up yet" |
| Output quality | not evaluated by a human listener yet |

Not tested: Linux, macOS, AMD GPUs, MP3 export of generated tracks, the optional `--lyrics` path with a real model.

## Install

Requires Python 3.10+ (3.11/3.12 recommended).

```
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[dev]"         # core + tests (synth backend only)
```

For the MusicGen backend, install the CUDA build of PyTorch first. A plain
`pip install torch` on Windows gives the CPU-only build:

```
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[musicgen]"
```

MP3 export needs `ffmpeg` on PATH. Run `aimusic doctor` to check your setup.
Model weights (about 2 GB for MusicGen-small) download from Hugging Face on first use.

## Usage

```
aimusic "30 sec calm piano, 70 bpm" --backend synth -y
aimusic "lofi study beats with rain" --backend musicgen --seed 42 --out song.mp3
aimusic "soundtrack with hip hop beats and punk kind of music at 450hz" --backend musicgen
aimusic "rainy night jazz" --llm llama3.2          # needs Ollama running locally
aimusic "epic cinematic trailer" --lyrics words.txt
aimusic doctor
```

| Flag | Meaning |
|---|---|
| `--backend` | `synth` (default), `musicgen`, `acestep` |
| `--duration` | seconds; overrides the prompt |
| `--seed` | reproducible results |
| `--tuning HZ` | concert A in Hz (default 440), applied after generation |
| `--lyrics FILE` | lyrics with `[verse]` / `[chorus]` tags |
| `--no-soundscape`, `--ambience-level` | control the rain/wind layer |
| `--llm MODEL` | enrich the style prompt with a local Ollama model |
| `--out` | output file, `.wav` or `.mp3` |
| `-y` | skip the confirmation prompt |

## Hardware notes

- MusicGen-small fits in 4 GB of VRAM. Larger models need more.
- On CPU everything works but neural generation is very slow.
- Different backends pin different torch/transformers versions. Use a separate
  virtual environment per backend.

## Licenses

- This code: MIT (see `LICENSE`).
- **MusicGen model weights are CC-BY-NC (non-commercial).** Check every model's
  license before commercial use. Rain/wind ambience is procedural.

## Roadmap

Based on a comparison of open models (from secondary sources plus ACE-Step's
official docs; not yet verified on this hardware):

1. Verify and wire up **ACE-Step 1.5**. Its README lists 10 s to 10 min per
   generation and its install guide lists 4 GB VRAM as the minimum (DiT-only mode),
   making it the only candidate that fits a 4 GB card and reaches 10 minutes.
2. Continuation for long tracks (feed the end of the audio back as the prompt),
   capped at 10 minutes, with progress, ETA and resume.
3. Gradio web UI.
4. CI and a GPU Docker image.

YuE, DiffRhythm and Stable Audio 3.0 were considered but need more VRAM than 4 GB.
