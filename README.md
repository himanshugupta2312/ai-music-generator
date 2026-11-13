# aimusic

Describe music in plain English; get an audio file. Runs locally with open-source models.

```
aimusic "Create 4 min long soothing music with rain in the background, good for studying, theme: sitting on a forest floor"
```

The tool shows how it understood your request, asks for confirmation, generates
the music, mixes in rain or wind if you asked for it, and saves a `.wav`/`.mp3`
plus a `.json` file with every setting used.

> **Early-stage project (v0.1).** One neural backend works (MusicGen). Tracks
> longer than 30 s are not yet continuous, and the ACE-Step backend is not
> implemented. Read [Limitations](#limitations) before relying on it.

## Try it in 2 minutes

```
git clone https://github.com/himanshugupta2312/ai-music-generator
cd ai-music-generator
python -m venv .venv
.venv\Scripts\activate            # Windows (use source .venv/bin/activate on Linux/macOS)
pip install -e ".[dev]"
aimusic "30 sec calm piano with rain" -y
```

That uses the built-in `synth` backend: no GPU, no downloads, but basic sound.
For real AI-generated music, install the [MusicGen backend](#install).

## Examples

Just write what you want. The tool pulls out duration, tempo, mood, genre,
ambience and tuning, then shows you the result before generating.

| You type | It understands | Listen |
|---|---|---|
| `aimusic "20 sec calm soothing piano music with rain for studying"` | 20 s, instrumental, calm piano style, **rain layer mixed in** | [calm-study-rain.mp3](examples/calm-study-rain.mp3) |
| `aimusic "20 sec punk rock with distorted guitars and fast drums"` | 20 s, punk rock, distorted guitars, energetic | [punk-rock.mp3](examples/punk-rock.mp3) |
| `aimusic "20 sec lofi hip hop beats, mellow, 85 bpm"` | 20 s, lo-fi hip hop, boom bap drums, 85 bpm | [lofi-beats.mp3](examples/lofi-beats.mp3) |

Each example has a `.json` next to it in [`examples/`](examples/) with the exact
settings and seed, so you can reproduce it.

> **About these samples:** they are the first output for each prompt, **not
> hand-picked**, and I have not listened to them for quality. Expect rough edges.
> They were made with MusicGen-small, whose weights are non-commercial
> (CC-BY-NC), so treat them as demonstrations only.

More things you can say:

```
aimusic "2 min mellow jazz, 70 bpm"                       # duration + tempo
aimusic "sleep music with wind, 1 hour"                    # wind layer (see limits for long tracks)
aimusic "soundtrack with hip hop beats and punk music at 450hz"   # tuning: A = 450 Hz
aimusic "epic cinematic trailer" --lyrics words.txt        # your own lyrics (backend support varies)
aimusic "rainy night jazz" --llm llama3.2                  # let a local Ollama model write the style prompt
aimusic "happy upbeat tune" --seed 42 --out tune.mp3       # reproducible, MP3 output
aimusic doctor                                             # check your setup
```

Recognised keywords include: calm, soothing, relax, sleep, study, focus, lofi,
forest, cinematic, epic, dark, happy, sad, jazz, piano, guitar, acoustic, punk,
hip hop, trap; plus rain and wind ambience. Anything else is passed through
as plain text, which the model may or may not understand.

## How it works

```
prompt -> parser -> settings -> backend -> ambience mix -> retune -> audio + .json
```

- **Parser** (rule-based): extracts duration ("4 min"), tempo ("120 bpm"), mood
  and genre keywords, rain/wind, instrumental vs vocals, and tuning ("450hz").
  Optionally a local Ollama model can rewrite the style description (`--llm`).
- **Backends** (swappable): `synth` (CPU, no model), `musicgen`, `acestep` (placeholder).
- **Ambience:** rain and wind are synthesized procedurally, so no sample licensing applies.
- **Tuning:** `--tuning 450` shifts finished audio so concert A sits at 450 Hz.

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
aimusic "lofi study beats" --backend musicgen -y
```

MP3 export needs `ffmpeg` on PATH. Run `aimusic doctor` to check your setup.
Model weights (about 2 GB for MusicGen-small) download from Hugging Face on first use.

### Flags

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

## Status

| Part | State |
|---|---|
| Plain-English parser | working, rule-based; 7 unit tests pass |
| `synth` backend (CPU, no model) | working; basic ambient pads, for testing and no-GPU use |
| `musicgen` backend | working. Tested on Windows 11, Quadro T2000 (4 GB), torch 2.14.1+cu126, transformers 5.22.2 |
| MP3 export | working (the three example files) |
| Rain/wind mixer, `--tuning`, `doctor` | working; the retune was verified on a pure tone, not on full tracks |
| `--llm` (Ollama) enrichment | implemented, **never run** against a live Ollama |
| `acestep` backend | **placeholder**; raises "not wired up yet" |
| Output quality | not evaluated by a human listener yet |

## Limitations

**Music quality and control**
- MusicGen-small is a small model. Genre mixes (for example hip hop + punk) may
  come out as generic rock with a beat rather than a real fusion.
- Instrumental only. MusicGen cannot sing, so `--lyrics` has no effect on it.
- Prompts are interpreted loosely. "Rain" or "punk" is a request, not a guarantee.
- Tuning (`--tuning`) is a post-process pitch shift. It also speeds the track up
  by the same small ratio (about 2% at 450 Hz) and uses simple interpolation.
  It cannot guarantee the final mix is in tune at that pitch.

**Length**
- MusicGen works in 30 s windows. Longer tracks are separate chunks joined by a
  crossfade, so the music **may change character** at the joins. There is no
  continuation yet and no 10-minute cap.

**Speed and hardware**
- Neural generation is slower than real time on a 4 GB laptop GPU (about 2.6x
  for MusicGen-small: a 20 s clip took 64-85 s here). A 4-minute track would
  take roughly 10 minutes; much longer jobs are not practical on that hardware.
- The `synth` backend is fast but sounds simple.
- Backends pin different torch/transformers versions, so use one virtual
  environment per backend.

**Testing and platforms**
- Only tested on one machine (Windows 11, NVIDIA Quadro T2000). Linux, macOS,
  AMD GPUs and CPU-only MusicGen are untested.
- The Ollama option has never been run.
- The parser is keyword-based and English-only; unusual phrasing may be missed.

**Licensing**
- This code is MIT. **MusicGen model weights are CC-BY-NC (non-commercial).**
  Check every model's license before commercial use.

## What's next

Planned in rough priority order. None of this is done yet.

1. **ACE-Step 1.5 backend.** Its README lists 10 s to 10 min per generation, and
   its install guide lists 4 GB VRAM as the minimum (DiT-only mode). That makes
   it the best candidate for full songs with vocals on a small GPU. This comes
   from docs and secondary sources; it has not been verified on this hardware.
2. **Continuous long tracks.** Feed the end of the audio back to the model as a
   prompt so the music continues instead of restarting, with a 10-minute cap,
   progress and ETA, and resume after a crash.
3. **Web UI** (Gradio).
4. **Better prompt understanding** (more genres, instruments and moods; tested LLM mode).
5. **CI and a GPU Docker image.**

YuE, DiffRhythm and Stable Audio 3.0 were considered but need more VRAM than a 4 GB card.

## License

MIT for the code (see [LICENSE](LICENSE)). Model weights keep their own licenses.
